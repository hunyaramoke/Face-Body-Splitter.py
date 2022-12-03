import gradio as gr

from modules import script_callbacks
import modules.shared as shared

import cv2
from anime_face_detector import create_detector
import glob
import os


def on_ui_tabs():
    with gr.Blocks() as fbs_interface:
        with gr.Row(equal_height=True):
            with gr.Column(variant='panel'):
                with gr.Column(variant='panel'):
                    input_dir = gr.Textbox(label="Input directory", placeholder="Input directory", value="")
                    output_dir = gr.Textbox(label="Output directory", placeholder="Output directory", value="")
                with gr.Column(variant='panel'):
                    padding_left_rate = gr.Slider(label="padding_left_rate", minimum=0.0, maximum=2.0, step=0.1, value=0.5 )
                    padding_bottom_rate= gr.Slider(label="padding_bottom_rate", minimum=0.0, maximum=2.0, step=0.1, value=0.1)
                    padding_right_rate = gr.Slider(label="padding_right_rate", minimum=0.0, maximum=2.0, step=0.1, value=0.5)
                    padding_top_rate = gr.Slider(label="padding_top_rate", minimum=0.0, maximum=2.0, step=0.1, value=0.8)
            with gr.Column(variant='panel'):
                status = gr.Textbox(label="", interactive=False, show_progress=True)
                
        
        dir_run = gr.Button(elem_id="dir_run", label="Generate", variant='primary')
        
        dir_run.click(
            fn=main,
            inputs=[input_dir, output_dir, padding_left_rate, padding_bottom_rate, padding_right_rate, padding_top_rate],
            outputs=[status]
        )

      
    return (fbs_interface, "Face Body Splitter", "fbs_interface"),


script_callbacks.on_ui_tabs(on_ui_tabs)


head_left   = 0
head_top    = 0
head_right  = 0
head_bottom = 0

face_width = 0
face_height = 0


# main
def getBody(image, padding_bottom_rate):
    height, width = image.shape[:2]

    body_left   = 0
    body_top    = (int)(head_bottom - (face_height * padding_bottom_rate))
    body_right  = width
    body_bottom = height

    sq_image = image[
        body_top : body_bottom,
        body_left : body_right
    ]

    return sq_image

def getHead(image, detector, padding_left_rate, padding_bottom_rate, padding_right_rate, padding_top_rate):
    global face_width
    global face_height
    global head_left
    global head_top
    global head_righ
    global head_bottom

    preds = detector(image)

    if len(preds) == 0:
        return None

    left = preds[0]['bbox'][0]
    bottom = preds[0]['bbox'][1]
    right = preds[0]['bbox'][2]
    top = preds[0]['bbox'][3]
    face_x = int((left + right) / 2)
    face_y = int((top + bottom) / 2)
    height, width = image.shape[:2]

    face_width  = (int)(right - left)
    face_height = (int)(top - bottom)

    head_left   = (int)(left - (face_width * padding_left_rate))
    head_top    = (int)(bottom - (face_height * padding_top_rate))
    head_right  = (int)(right + (face_width * padding_right_rate))
    head_bottom = (int)(top + (face_height * padding_bottom_rate))

    if head_left < 0:
        head_left = 0
    if head_top < 0:
        head_top = 0
    if head_right > width:
        head_right = width
    if head_bottom > height:
        head_bottom = height

    sq_image = image[
        head_top : head_bottom,
        head_left : head_right
    ]

    return sq_image



def main(input_dir, output_dir, padding_left_rate, padding_bottom_rate, padding_right_rate, padding_top_rate):

    # load model
    detector = create_detector('yolov3')
    
    output_extension = "png"#@param{type:"string"}

    if not os.path.exists(output_dir):
      raise ValueError("output_dir is not exist")

    paths = glob.glob(input_dir + "/*")
    paths_len = len(paths)
    error_list = []

    for i, path in enumerate(paths):
        print(f"{i+1}/{paths_len} : {path}")
        basename = os.path.split(path)[1].split(".")[0]

        image = cv2.imread(path)
        head_image = getHead(image, detector, padding_left_rate, padding_bottom_rate, padding_right_rate, padding_top_rate)
        body_image = getBody(image, padding_bottom_rate)

        # Skip if face does not exist
        if head_image is None:
            print("Could not recognize the face in this image")
            error_list.append(path)
            continue
        # Skip if face does not exist
        if body_image is None:
            print("Could not recognize the face in this image")
            error_list.append(path)
            continue
          
        #画像の書き出し
        cv2.imwrite(f"{output_dir}/{basename}_head.{output_extension}", head_image)
        cv2.imwrite(f"{output_dir}/{basename}_body.{output_extension}", body_image)

    print("Split_finished")
    return "Split finished"