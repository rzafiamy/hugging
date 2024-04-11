import argparse
from huggingface_hub import hf_hub_download

def download_model_from_huggingface(model_name, model_file, hf_token=None, local_dir='models'):
    """
    Downloads a model from Hugging Face hub.

    Args:
    - model_name: Name of the model to download.
    - model_file: The specific file to download.
    - hf_token: Hugging Face API token for authentication.
    - local_dir: Local directory to save the downloaded file.
    """
    model_path = hf_hub_download(model_name,
                                 filename=model_file,
                                 local_dir=local_dir,
                                 use_auth_token=hf_token)
    print("My model path: ", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download a model from Hugging Face hub.')
    parser.add_argument('--model', type=str, help='Name of the model to download', required=True)
    parser.add_argument('--file', type=str, help='The specific file to download', required=True)
    parser.add_argument('--token', type=str, default=None, help='Hugging Face API token (optional)')
    parser.add_argument('--local_dir', type=str, default='models', help='Local directory to save the model (default: models)')

    args = parser.parse_args()

    download_model_from_huggingface(args.model, args.file, args.token, args.local_dir)
