# Import required libraries 
import argparse
import sys
import os
sys.path.append('../')
sys.path.append('../source_code')
sys.path.append('../source_code/distancing/')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import time
import sys
import math
import platform
from common.bus_call import bus_call
from common.FPS import GETFPS
import pyds



PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

g_args=None
# Define variables to be used later
fps_streams={}
pgie_classes_str= ["Vehicle", "TwoWheeler", "Person","RoadSign"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sources", type=int, default=1, help="Number of sources, it replicates inputs if its is greater than length of inputs")
    parser.add_argument("--prof", type=bool, default=False, help="Profiling Mode , Profiles with a shorter video clip.")
    args = parser.parse_args()

    return args


    
def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)   

def create_source_bin(args, index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write("Failed to add ghost pad in source bin \n")
        return None
    return nbin


## Make Element or Print Error and any other detail
def make_elm_or_print_err(factoryname, name, printedname, detail=""):
  print("Creating", printedname)
  elm = Gst.ElementFactory.make(factoryname, name)
  if not elm:
     sys.stderr.write("Unable to create " + printedname + " \n")
  if detail:
     sys.stderr.write(detail)
  return elm

############# Define Computation required for our pipeline #################

def compute_dist(p1, p2):
    
    (x1, y1, h1) = p1;
    (x2, y2, h2) = p2;
    dx = x2 - x1;
    dy = y2 - y1;

    lx = dx * 170 * (1/h1 + 1/h2) / 2;
    ly = dy * 170 * (1/h1 + 1/h2) / 2;

    l = math.sqrt(lx*lx + ly*ly);
    return l


def get_min_distances(centroids):
    mini=[]
    for i in range(len(centroids)):
        distance=[]
        for j in range(len(centroids)):
            distance.append(compute_dist(centroids[i],centroids[j]))
        distance[i]=10000000
        mini.append(min(distance))
    return mini


def visualize(objs):
    violations = 0 
    dist_threshold = 160  # Distance in cms
    for obj in objs:
        min_dist = obj["min_dist"]
        redness_factor = 1.5
        obj["violated"] = (min_dist < dist_threshold)
        violations = violations + int(min_dist < dist_threshold)
    return violations

def get_centroid(rect):

    xmin = rect.left
    xmax = rect.left + rect.width
    ymin = rect.top
    ymax = rect.top + rect.height
    centroid_x = (xmin + xmax) / 2
    centroid_y = (ymin + ymax) / 2

    return (centroid_x, centroid_y, rect.height)

def compute_min_distances_cpp(objs):
    centroids = [o["centroid"] for o in objs]    
    min_distances = get_min_distances(centroids)
    for o in range(len(objs)):
        objs[o]["min_dist"] = min_distances[o]



############## Working with the Metadata ################

def src_pad_buffer_probe(pad,info,u_data):
        # Set frame_number & rectangles to draw as 0 
    frame_number=0
    num_rects=0
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        objects=[]
        # Get frame number , number of rectables to draw and object metadata
        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        #Intiallizing object counter with 0.
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE:0,
            PGIE_CLASS_ID_PERSON:0,
            PGIE_CLASS_ID_BICYCLE:0,
            PGIE_CLASS_ID_ROADSIGN:0
        }
        
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            # Increment Object class by 1 and Set Box border to Red color     
            obj_counter[obj_meta.class_id] +=1
            obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.0)
            
            if (obj_meta.class_id == PGIE_CLASS_ID_PERSON):
                obj = {}
                obj["tracker_id"] = obj_meta.object_id
                obj["unique_id"] = obj_meta.unique_component_id
                obj["centroid"] = get_centroid(obj_meta.rect_params)
                obj["obj_meta"] = obj_meta
                objects.append(obj)
            else:
                obj_meta.rect_params.border_width = 0

            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        # Get the number of violations 
        compute_min_distances_cpp(objects)
        violations = visualize(objects)
    
        print("Frame Number={} Number of Objects={} Vehicle_count={} Person_count={} Violations={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON],violations))
        
        # Get frame rate through this probe
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


def main():
    
    args = parse_args()
    global g_args
    g_args = args
    num_sources = args.num_sources
    path = os.path.abspath(os.getcwd())
    if (args.prof):
        INPUT_VIDEO = 'file://' + path +'/../source_code/dataset/wt_prof.mp4'
    else :
        INPUT_VIDEO = 'file://' + path +'/../source_code/dataset/wt.mp4'
    
    print("Creating pipeline with "+str(num_sources)+" streams")
    # Initialise FPS
    for i in range(0,num_sources):
            fps_streams["stream{0}".format(i)]=GETFPS(i)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    ########### Create Elements required for the Pipeline ########### 

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = make_elm_or_print_err("nvstreammux", "Stream-muxer","Stream-muxer") 

    pipeline.add(streammux)

    for i in range(num_sources):
        print("Creating source_bin ",i," \n ")
        uri_name=INPUT_VIDEO
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin(args, i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad = streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)


    # Use nvinfer to run inferencing on decoder's output, behaviour of inferencing is set through config file
    pgie = make_elm_or_print_err("nvinfer", "primary-inference" ,"pgie")
    # Create Sink for storing the output 
    fakesink = make_elm_or_print_err("fakesink", "fakesink", "Sink")

    # Queues to enable buffering
    queue1=make_elm_or_print_err("queue","queue1","queue1")
    queue2=make_elm_or_print_err("queue","queue2","queue2")
    queue3=make_elm_or_print_err("queue","queue3","queue3")

    ############ Set properties for the Elements ############
    # Set Input Width , Height and Batch Size 
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', num_sources)
    # Timeout in microseconds to wait after the first buffer is available 
    # to push the batch even if a complete batch is not formed.
    streammux.set_property('batched-push-timeout', 4000000)
    # Set configuration file for nvinfer 
    pgie.set_property('config-file-path', "../source_code/N3/dstest1_pgie_config.txt")
    # Setting batch_size for pgie
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != num_sources):
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", num_sources," \n")
        pgie.set_property("batch-size",num_sources)
    # Fake sink properties 
    fakesink.set_property("sync", 0)
    fakesink.set_property("async", 0)

    ########## Add and Link ELements in the Pipeline ########## 

    print("Adding elements to Pipeline \n")
    pipeline.add(queue1)
    pipeline.add(pgie)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(fakesink)

    print("Linking elements in the Pipeline \n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(queue3)
    queue3.link(fakesink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    print("Added and Linked elements to pipeline")

    src_pad=queue3.get_static_pad("src")
    if not src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        src_pad.add_probe(Gst.PadProbeType.BUFFER, src_pad_buffer_probe, 0)

    # List the sources
    print("Now playing...")
    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    start_time = time.time()
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    sys.exit(main())
