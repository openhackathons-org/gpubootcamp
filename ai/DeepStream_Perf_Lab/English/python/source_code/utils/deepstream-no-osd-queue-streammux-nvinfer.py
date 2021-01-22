# Import required libraries 
import argparse
import sys
import os
sys.path.append('../')
sys.path.append('../source_code')
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
fps_streams_new={}
pgie_classes_str= ["Vehicle", "TwoWheeler", "Person","RoadSign"]

################ Three Stream Pipeline ###########
# Define Input and output Stream information 
INPUT_VIDEO = 'file:///opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sources", type=int, default=1, help="Number of sources, it replicates inputs if its is greater than length of inputs")
    parser.add_argument("--sgie-batch-size", type=int, default=1, help="Batch size for Seconday inference")
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
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
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

# src_pad_buffer_probe  
def src_pad_buffer_probe(pad,info,u_data):
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BICYCLE:0,
        PGIE_CLASS_ID_ROADSIGN:0
    }
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
        
        # Get frame number , number of rectables to draw and object metadata
        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            # Increment Object class by 1     
            obj_counter[obj_meta.class_id] += 1
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        
        print("Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON]))
        
        # FPS Probe      
        fps_streams_new["stream{0}".format(frame_meta.pad_index)].get_fps()

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
    sgie_batch_size = args.sgie_batch_size
    path = os.path.abspath(os.getcwd())
    
    if (args.prof):
        INPUT_VIDEO = 'file://' + path +'/../source_code/dataset/sample_720p_prof.mp4'
    else :
        INPUT_VIDEO = 'file:///opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264'
    print("Creating pipeline with "+str(num_sources)+" streams")
    # Initialise FPS
    for i in range(0,num_sources):
            fps_streams_new["stream{0}".format(i)]=GETFPS(i)

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
    # Use nvtracker to give objects unique-ids
    tracker = make_elm_or_print_err("nvtracker", "tracker",'tracker')
    # Seconday inference for Finding Car Color
    sgie1 = make_elm_or_print_err("nvinfer", "secondary1-nvinference-engine",'sgie1')
    # Seconday inference for Finding Car Make
    sgie2 = make_elm_or_print_err("nvinfer", "secondary2-nvinference-engine",'sgie2')
    # # Seconday inference for Finding Car Type
    sgie3 = make_elm_or_print_err("nvinfer", "secondary3-nvinference-engine",'sgie3')
    # Create Sink for storing the output 
    fakesink = make_elm_or_print_err("fakesink", "fakesink", "Sink")

    # Queues to enable buffering
    queue1=make_elm_or_print_err("queue","queue1","queue1")
    queue2=make_elm_or_print_err("queue","queue2","queue2")
    queue3=make_elm_or_print_err("queue","queue3","queue3")
    queue4=make_elm_or_print_err("queue","queue4","queue4")
    queue5=make_elm_or_print_err("queue","queue5","queue5")
    queue6=make_elm_or_print_err("queue","queue6","queue6")

    ############ Set properties for the Elements ############
    # Set Input Width , Height and Batch Size 
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', num_sources)
    # Timeout in microseconds to wait after the first buffer is available 
    # to push the batch even if a complete batch is not formed.
    streammux.set_property('batched-push-timeout', 4000000)
    # Set configuration file for nvinfer 
    pgie.set_property('config-file-path', "../source_code/N1/dstest4_pgie_config.txt")
    sgie1.set_property('config-file-path', "../source_code/N1/dstest4_sgie1_config.txt")
    sgie2.set_property('config-file-path', "../source_code/N1/dstest4_sgie2_config.txt")
    sgie3.set_property('config-file-path', "../source_code/N1/dstest4_sgie3_config.txt")
    # Setting batch_size for pgie
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != num_sources):
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", num_sources," \n")
        pgie.set_property("batch-size",num_sources)
        
    #################### Secondary Batch size ######################
     # Setting batch_size for sgie1
    sgie1_batch_size=sgie1.get_property("batch-size")
    if(sgie1_batch_size != sgie_batch_size):
        print("WARNING: Overriding infer-config batch-size",sgie1_batch_size," with number of sources ", sgie_batch_size," \n")
        sgie1.set_property("batch-size",sgie_batch_size)
    # Setting batch_size for sgie2
    sgie2_batch_size=sgie2.get_property("batch-size")
    if(sgie2_batch_size != sgie_batch_size):
        print("WARNING: Overriding infer-config batch-size",sgie2_batch_size," with number of sources ", sgie_batch_size," \n")
        sgie2.set_property("batch-size",sgie_batch_size)
    # Setting batch_size for sgie2
    sgie3_batch_size=sgie3.get_property("batch-size")
    if(sgie3_batch_size != sgie_batch_size):
        print("WARNING: Overriding infer-config batch-size",sgie3_batch_size," with number of sources ", sgie_batch_size," \n")
        sgie3.set_property("batch-size",sgie_batch_size)
    #Set properties of tracker from tracker_config
    config = configparser.ConfigParser()
    config.read('../source_code/N1/dstest4_tracker_config.txt')
    config.sections()
    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)

    # Fake sink properties 
    fakesink.set_property("sync", 0)
    fakesink.set_property("async", 0)

    ########## Add and Link ELements in the Pipeline ########## 

    print("Adding elements to Pipeline \n")
    pipeline.add(queue1)
    pipeline.add(pgie)
    pipeline.add(queue2)
    pipeline.add(tracker)
    pipeline.add(queue3)
    pipeline.add(sgie1)
    pipeline.add(queue4)
    pipeline.add(sgie2)
    pipeline.add(queue5)
    pipeline.add(sgie3)
    pipeline.add(queue6)
    pipeline.add(fakesink)

    print("Linking elements in the Pipeline \n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tracker)
    tracker.link(queue3)
    queue3.link(sgie1)
    sgie1.link(queue4)
    queue4.link(sgie2)
    sgie2.link(queue5)
    queue5.link(sgie3)
    sgie3.link(fakesink)
    # queue6.link(fakesink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    print("Added and Linked elements to pipeline")

    src_pad=sgie3.get_static_pad("src")
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
