import argparse
import pathlib
from s2s import S2STransformer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="Path to pl model",
    required=True,
)

parser.add_argument("--output_dir", type=str, required=True, help="Where to store the final model.")


args = parser.parse_args()
model_name_or_path = args.model_name_or_path
output_dir = args.output_dir
model_output_name = pathlib.Path(output_dir).as_posix()
# config_output_name = pathlib.Path(output_dir, "config").as_posix()
tokenizer_output_name = pathlib.Path(output_dir, "tokenizer").as_posix()


model = S2STransformer.load_from_checkpoint(model_name_or_path)
# print(model)
model.model.save_pretrained(model_output_name)
# model.config.save_pretrained(config_output_name)
model.tokenizer.save_pretrained(tokenizer_output_name)