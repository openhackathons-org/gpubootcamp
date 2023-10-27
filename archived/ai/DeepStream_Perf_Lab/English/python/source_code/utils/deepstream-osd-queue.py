# Import required libraries 
import sys
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


# Define variables to be used later
fps_streams={}

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080

TILED_OUTPUT_WIDTH=1920
TILED_OUTPUT_HEIGHT=1080
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 0
pgie_classes_str= ["Vehicle", "TwoWheeler", "Person","RoadSign"]

################ Three Stream Pipeline ###########
# Define Input and output Stream information 
num_sources = 3 
INPUT_VIDEO_1 = '/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.h264'
INPUT_VIDEO_2 = '/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.h264'
INPUT_VIDEO_3 = '/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.h264'
OUTPUT_VIDEO_NAME = "../source_code/N1/ds_out2.mp4"


## Make Element or Print Error and any other detail
def make_elm_or_print_err(factoryname, name, printedname, detail=""):
  print("Creating", printedname)
  elm = Gst.ElementFactory.make(factoryname, name)
  if not elm:
     sys.stderr.write("Unable to create " + printedname + " \n")
  if detail:
     sys.stderr.write(detail)
  return elm


# #### Initialise GStreamer and Create an Empty Pipeline

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


# #### Create Elements that are required for our pipeline



########### Create Elements required for the Pipeline ########### 

######### Defining Stream 1 
# Source element for reading from the file
source1 = make_elm_or_print_err("filesrc", "file-source-1",'file-source-1')
# Since the data format in the input file is elementary h264 stream,we need a h264parser
h264parser1 = make_elm_or_print_err("h264parse", "h264-parser-1","h264-parser-1")
# Use nvdec_h264 for hardware accelerated decode on GPU
decoder1 = make_elm_or_print_err("nvv4l2decoder", "nvv4l2-decoder-1","nvv4l2-decoder-1")
   
##########

########## Defining Stream 2 
# Source element for reading from the file
source2 = make_elm_or_print_err("filesrc", "file-source-2","file-source-2")
# Since the data format in the input file is elementary h264 stream, we need a h264parser
h264parser2 = make_elm_or_print_err("h264parse", "h264-parser-2", "h264-parser-2")
# Use nvdec_h264 for hardware accelerated decode on GPU
decoder2 = make_elm_or_print_err("nvv4l2decoder", "nvv4l2-decoder-2","nvv4l2-decoder-2")
########### 

########## Defining Stream 3
# Source element for reading from the file
source3 = make_elm_or_print_err("filesrc", "file-source-3","file-source-3")
# Since the data format in the input file is elementary h264 stream, we need a h264parser
h264parser3 = make_elm_or_print_err("h264parse", "h264-parser-3", "h264-parser-3")
# Use nvdec_h264 for hardware accelerated decode on GPU
decoder3 = make_elm_or_print_err("nvv4l2decoder", "nvv4l2-decoder-3","nvv4l2-decoder-3")
########### 
    
# Create nvstreammux instance to form batches from one or more sources.
streammux = make_elm_or_print_err("nvstreammux", "Stream-muxer","Stream-muxer") 
# Use nvinfer to run inferencing on decoder's output, behaviour of inferencing is set through config file
pgie = make_elm_or_print_err("nvinfer", "primary-inference" ,"pgie")
# Use nvtracker to give objects unique-ids
tracker = make_elm_or_print_err("nvtracker", "tracker",'tracker')
# Seconday inference for Finding Car Color
sgie1 = make_elm_or_print_err("nvinfer", "secondary1-nvinference-engine",'sgie1')
# Seconday inference for Finding Car Make
sgie2 = make_elm_or_print_err("nvinfer", "secondary2-nvinference-engine",'sgie2')
# Seconday inference for Finding Car Type
sgie3 = make_elm_or_print_err("nvinfer", "secondary3-nvinference-engine",'sgie3')
# Creating Tiler to present more than one streams
tiler=make_elm_or_print_err("nvmultistreamtiler", "nvtiler","nvtiler")
# Use convertor to convert from NV12 to RGBA as required by nvosd
nvvidconv = make_elm_or_print_err("nvvideoconvert", "convertor","nvvidconv")
# Create OSD to draw on the converted RGBA buffer
nvosd = make_elm_or_print_err("nvdsosd", "onscreendisplay","nvosd")
# Creating queue's to buffer incoming data from pgie
queue1=make_elm_or_print_err("queue","queue1","queue1")
# Creating queue's to buffer incoming data from tiler
queue2=make_elm_or_print_err("queue","queue2","queue2")
# Creating queue's to buffer incoming data from nvvidconv
queue3=make_elm_or_print_err("queue","queue3","queue3")
# Creating queue's to buffer incoming data from nvosd
queue4=make_elm_or_print_err("queue","queue4","queue4")
# Creating queue's to buffer incoming data from nvvidconv2
queue5=make_elm_or_print_err("queue","queue5","queue5")
# Creating queue's to buffer incoming data from nvtracker
queue6=make_elm_or_print_err("queue","queue6","queue6")
# Creating queue's to buffer incoming data from sgie1
queue7=make_elm_or_print_err("queue","queue7","queue7")
# Creating queue's to buffer incoming data from sgie2
queue8=make_elm_or_print_err("queue","queue8","queue8")
# Creating queue's to buffer incoming data from sgie3
queue9=make_elm_or_print_err("queue","queue9","queue9")
# Use convertor to convert from NV12 to RGBA as required by nvosd
nvvidconv2 = make_elm_or_print_err("nvvideoconvert", "convertor2","nvvidconv2")
# Place an encoder instead of OSD to save as video file
encoder = make_elm_or_print_err("avenc_mpeg4", "encoder", "Encoder")
# Parse output from Encoder 
codeparser = make_elm_or_print_err("mpeg4videoparse", "mpeg4-parser", 'Code Parser')
# Create a container
container = make_elm_or_print_err("qtmux", "qtmux", "Container")
# Create Sink for storing the output 
sink = make_elm_or_print_err("filesink", "filesink", "Sink")


############ Set properties for the Elements ############
# Set Input Video files 
source1.set_property('location', INPUT_VIDEO_1)
source2.set_property('location', INPUT_VIDEO_2)
source3.set_property('location', INPUT_VIDEO_2)
# Set Input Width , Height and Batch Size 
streammux.set_property('width', 1920)
streammux.set_property('height', 1080)
streammux.set_property('batch-size', num_sources)
# Timeout in microseconds to wait after the first buffer is available 
# to push the batch even if a complete batch is not formed.
streammux.set_property('batched-push-timeout', 4000000)
# Set configuration file for nvinfer 
# Set Congifuration file for nvinfer 
pgie.set_property('config-file-path', "../source_code/N1/dstest4_pgie_config.txt")
sgie1.set_property('config-file-path', "../source_code/N1/dstest4_sgie1_config.txt")
sgie2.set_property('config-file-path', "../source_code/N1/dstest4_sgie2_config.txt")
sgie3.set_property('config-file-path', "../source_code/N1/dstest4_sgie3_config.txt")
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
        
    
# Set display configurations for nvmultistreamtiler    
tiler_rows=int(2)
tiler_columns=int(2)
tiler.set_property("rows",tiler_rows)
tiler.set_property("columns",tiler_columns)
tiler.set_property("width", TILED_OUTPUT_WIDTH)
tiler.set_property("height", TILED_OUTPUT_HEIGHT)

# Set encoding properties and Sink configs
encoder.set_property("bitrate", 2000000)
sink.set_property("location", OUTPUT_VIDEO_NAME)
sink.set_property("sync", 0)
sink.set_property("async", 0)


# We now link all the elements in the order we prefer and create Gstreamer bus to feed all messages through it. 

########## Add and Link ELements in the Pipeline ########## 

print("Adding elements to Pipeline \n")
pipeline.add(source1)
pipeline.add(h264parser1)
pipeline.add(decoder1)
pipeline.add(source2)
pipeline.add(h264parser2)
pipeline.add(decoder2)
pipeline.add(source3)
pipeline.add(h264parser3)
pipeline.add(decoder3)
pipeline.add(streammux)
pipeline.add(pgie)
pipeline.add(tracker)
pipeline.add(sgie1)
pipeline.add(sgie2)
pipeline.add(sgie3)
pipeline.add(tiler)
pipeline.add(nvvidconv)
pipeline.add(nvosd)
pipeline.add(queue1)
pipeline.add(queue2)
pipeline.add(queue3)
pipeline.add(queue4)
pipeline.add(queue5)
pipeline.add(queue6)
pipeline.add(queue7)
pipeline.add(queue8)
pipeline.add(queue9)
pipeline.add(nvvidconv2)
pipeline.add(encoder)
pipeline.add(codeparser)
pipeline.add(container)
pipeline.add(sink)

print("Linking elements in the Pipeline \n")

source1.link(h264parser1)
h264parser1.link(decoder1)


###### Create Sink pad and connect to decoder's source pad 
sinkpad1 = streammux.get_request_pad("sink_0")
if not sinkpad1:
    sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
srcpad1 = decoder1.get_static_pad("src")
if not srcpad1:
    sys.stderr.write(" Unable to get source pad of decoder \n")
    
srcpad1.link(sinkpad1)

######

###### Create Sink pad and connect to decoder's source pad 
source2.link(h264parser2)
h264parser2.link(decoder2)

sinkpad2 = streammux.get_request_pad("sink_1")
if not sinkpad2:
    sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
srcpad2 = decoder2.get_static_pad("src")
if not srcpad2:
    sys.stderr.write(" Unable to get source pad of decoder \n")
    
srcpad2.link(sinkpad2)

######

###### Create Sink pad and connect to decoder's source pad 
source3.link(h264parser3)
h264parser3.link(decoder3)

sinkpad3 = streammux.get_request_pad("sink_2")
if not sinkpad2:
    sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
srcpad3 = decoder3.get_static_pad("src")
if not srcpad3:
    sys.stderr.write(" Unable to get source pad of decoder \n")
    
srcpad3.link(sinkpad3)

######


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
sgie3.link(queue6)
queue6.link(tiler)
tiler.link(queue7)
queue7.link(nvvidconv)
nvvidconv.link(queue8)
queue8.link(nvosd)
nvosd.link(queue9)
queue9.link(nvvidconv2)
nvvidconv2.link(encoder)
encoder.link(codeparser)
codeparser.link(container)
container.link(sink)


# create an event loop and feed gstreamer bus mesages to it
loop = GObject.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect ("message", bus_call, loop)



# tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
# and update params for drawing rectangle, object information etc.
def tiler_src_pad_buffer_probe(pad,info,u_data):
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
            # Increment Object class by 1 and Set Box border to Red color     
            obj_counter[obj_meta.class_id] += 1
            obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.0)
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        ################## Setting Metadata Display configruation ############### 
        # Acquiring a display meta object.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])
        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # Set(red, green, blue, alpha); Set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # Set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string to print in notebook
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        
        ############################################################################
        # Get frame rate through this probe
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


tiler_src_pad=sgie3.get_static_pad("src")
if not tiler_src_pad:
    sys.stderr.write(" Unable to get src pad \n")
else:
    tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_src_pad_buffer_probe, 0)


# Now with everything defined , we can start the playback and listen the events.

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
print("Exiting app\n")
pipeline.set_state(Gst.State.NULL)
print("--- %s seconds ---" % (time.time() - start_time))


