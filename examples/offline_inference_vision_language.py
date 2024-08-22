"""
This example shows how to use vLLM for running offline inference 
with the correct prompt format on vision language models.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser

# Input image and question
cherry_blossom_image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
stop_sign_image = ImageAsset("stop_sign").pil_image.convert("RGB")

# LLaVA-1.6/LLaVA-NeXT
def run_llava_next():

    llm = LLM(model="/models/llava-v1.6-mistral-7b-hf",tensor_parallel_size=2)
    inputs = [
            # {
            #     "prompt": f"[INST] What is the content of this image? [/INST]",
            # },
            # {
            #     "prompt": f"[INST] <image>\nWhat is the content of this image? [/INST]",
            #     "multi_modal_data": {
            #         "image": cherry_blossom_image
            #     }
            # },
            # {
            #     "prompt": f"[INST] <image>\nWhat is the content of this image? [/INST]",
            #     "multi_modal_data": {
            #         "image": stop_sign_image
            #     }
            # },
            {
                "prompt": f"[INST] <image> \n <image> \n请用中文描述这两幅图片的内容。[/INST]",
                "multi_modal_data": {
                    "image": [cherry_blossom_image, stop_sign_image]
                }
            },
        ]
    return llm, inputs



def run_GlintCom_03():
    # system="<|im_start|>system\nYou are a helpful assistant."
    llm = LLM(model="/models/llava-7b-unicom-qwen2-mul-hd-cappre-1m",tensor_parallel_size=2)

    inputs = [
    # {
    #     "prompt": f"USER: What is the content of this image?\nASSISTANT:",
    # },
    # {
    #     "prompt": f"USER: <image>\nWhat is the content of this image?\nASSISTANT:",
    #     "multi_modal_data": {
    #         "image": cherry_blossom_image
    #     }
    # },
    {
        "prompt": f"<|im_start|>user: <image>\n请用中文描述这图片的内容。<|im_end|>\n<|im_start|>assistant\n",
        "multi_modal_data": {
            "image": stop_sign_image
        }
    },
    {
        "prompt": f"<|im_start|>user: <image> \n <image> \n请用中文描述这两幅图片的内容。<|im_end|>\n<|im_start|>assistant\n",
        "multi_modal_data": {
            "image": [stop_sign_image, cherry_blossom_image]
            # "image": [cherry_blossom_image, stop_sign_image]
        }
    },
    ]
    return llm, inputs


model_example_map = {
    "llava-next": run_llava_next,
    "glintcom-03": run_GlintCom_03,
}


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    llm, inputs = model_example_map[model]()

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(max_tokens=2048,top_p=0.95,temperature=0)

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for i, o in zip(inputs, outputs):
        generated_text = o.outputs[0].text
        print(i["prompt"], generated_text)

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="llava",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')

    args = parser.parse_args()
    main(args)
