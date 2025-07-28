import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform
import glob
from tqdm import tqdm


def process_single_image(input_image_path, model, T_val, args):
    """
    Process a single image through the model.
    
    Args:
        input_image_path (str): Path to the input image
        model: The loaded CycleGAN_Turbo model
        T_val: The image transformation pipeline
        args: Command line arguments
        
    Returns:
        tuple: (output_pil, original_filename) or (None, None) if processing fails
    """
    try:
        input_image = Image.open(input_image_path).convert('RGB')
        
        # translate the image
        with torch.no_grad():
            input_img = T_val(input_image)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
            if args.use_fp16:
                x_t = x_t.half()
            output = model(x_t, direction=args.direction, caption=args.prompt)

        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)
        
        original_filename = os.path.basename(input_image_path)
        return output_pil, original_filename
        
    except Exception as e:
        print(f"Error processing {input_image_path}: {str(e)}")
        return None, None


def get_image_files(directory):
    """
    Get all image files from a directory.
    
    Args:
        directory (str): Path to the directory
        
    Returns:
        list: List of image file paths
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
        image_files.extend(glob.glob(os.path.join(directory, ext.upper())))
    
    return sorted(image_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, help='path to the input image')
    parser.add_argument('--input_dir', type=str, help='path to the directory containing input images')
    parser.add_argument('--prompt', type=str, required=False, help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_name', type=str, default=None, help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method')
    parser.add_argument('--direction', type=str, default=None, help='the direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    args = parser.parse_args()

    # Validate input arguments
    if args.input_image is None and args.input_dir is None:
        raise ValueError('Either --input_image or --input_dir must be provided')
    
    if args.input_image is not None and args.input_dir is not None:
        raise ValueError('Only one of --input_image or --input_dir should be provided')

    # only one of model_name and model_path should be provided
    if args.model_name is None != args.model_path is None:
        raise ValueError('Either model_name or model_path should be provided')

    if args.model_path is not None and args.prompt is None:
        raise ValueError('prompt is required when loading a custom model_path.')

    if args.model_name is not None:
        assert args.prompt is None, 'prompt is not required when loading a pretrained model.'
        assert args.direction is None, 'direction is not required when loading a pretrained model.'

    # initialize the model
    model = CycleGAN_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    if args.use_fp16:
        model.half()

    T_val = build_transform(args.image_prep)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process single image
    if args.input_image is not None:
        output_pil, filename = process_single_image(args.input_image, model, T_val, args)
        if output_pil is not None:
            output_pil.save(os.path.join(args.output_dir, filename))
            print(f"Processed: {args.input_image} -> {os.path.join(args.output_dir, filename)}")
    
    # Process directory of images
    elif args.input_dir is not None:
        if not os.path.isdir(args.input_dir):
            raise ValueError(f"Input directory does not exist: {args.input_dir}")
        
        image_files = get_image_files(args.input_dir)
        if not image_files:
            raise ValueError(f"No image files found in directory: {args.input_dir}")
        
        print(f"Found {len(image_files)} images to process")
        
        successful = 0
        failed = 0
        
        for image_path in tqdm(image_files, desc="Processing images"):
            output_pil, filename = process_single_image(image_path, model, T_val, args)
            
            if output_pil is not None:
                output_pil.save(os.path.join(args.output_dir, filename))
                successful += 1
            else:
                failed += 1
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful} images")
        print(f"Failed to process: {failed} images")
        print(f"Output saved to: {args.output_dir}")
