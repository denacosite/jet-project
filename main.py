import sys

sys.path.append('functions')
import funs

if __name__ == "__main__":

    program_description = [
        'Shot -> mold: -1',
        'Show -> mold image: 0',
        'Shot -> object: 1',
        'Live: 2',
        'Subtraction: 3'
        ]

    result_description = ''
    for text in program_description:
        result_description += text + '\n'
    
    input_opr = input(result_description)

    if(input_opr == 1):
        funs.start_shoting_from_object()

    if(input_opr == -1):
        funs.shot_from_mold()

    if (input_opr == 0):
        funs.show_mold()

    if (input_opr == 2):
        funs.live_camera_without_capture()

    if (input_opr == 3):
        funs.start_subtraction()

    if (input_opr == 5):
        funs.otsu_thresholding()

    if (input_opr == 6):
        funs.img2double()