import argparse
import parse
from pathlib import Path

_PREDICT_FILE = 'lxmert/src/tasks/lxmert_yolov5s_predict.py'
_MODELING_FILE = '/home/tx2/work/AcceleratorBERT/lxmert/src/lxrt/modeling.py'
components = {
    'real_run': _PREDICT_FILE,
    'preprocess_image': _PREDICT_FILE,
    'run_detection': _PREDICT_FILE,
    'postprocess_feature': _PREDICT_FILE,
    'image_encoder': _MODELING_FILE,
    'ques_encoder': _MODELING_FILE,
    'cross_encoder': _MODELING_FILE,

}


def main(args):
    input_file = Path(args.input_file)
    lines = input_file.read_text()
    breakdowns = {k: next(parse.findall(
        f"{k} ({v}) ({{:n}} samples", lines))[0] for k, v in components.items()}
    for k, v in breakdowns.items():
        print(f'{k}, {v}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str,
                        help="input svg file")
    main(parser.parse_args())
