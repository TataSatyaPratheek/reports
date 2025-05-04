import nltk

s_to_download = ["punkt", "wordnet", "stopwords"]
download_successful = True

for resource in s_to_download:
    try:
        nltk.download(resource, quiet=True)
        # Optionally, verify the resource is now available
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except Exception as e:
        print(f"Failed to download or verify NLTK resource '{resource}': {e}")
        download_successful = False

print("All downloads successful:", download_successful)
