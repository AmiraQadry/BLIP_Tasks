# Bootstrapping Language-Image Pre-training (BLIP) Tasks
![](https://www.marktechpost.com/wp-content/uploads/2022/03/Screen-Shot-2022-03-01-at-11.34.30-PM.png)

## Introduction
This project demonstrates tasks related to Bootstrapping Language-Image Pre-training (BLIP). 
The objectives are to set up the environment, load the model, preprocess data, train the model, and evaluate its performance.

## Objectives
- Set up the environment and install necessary packages
- Load the BLIP model
- Preprocess data
- Train the model
- Evaluate the model's performance
- Summarize the findings and suggest improvements

## Environment Setup
In this section, we install the required packages and set up the environment for running the BLIP tasks.

```python
# Install necessary packages
!pip install transformers
!pip install torch
```

## Model Loading and Preprocessing
Here, we load the BLIP model and perform necessary preprocessing steps on the data.

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt

# Load the processor and the model
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
```

## Functions

### Display Image

This function displays an image from a URL with an optional title.

```python
def display_image(image_url, title=None):
    image = Image.open(requests.get(image_url, stream=True).raw)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
```

### Softmax Function

This function applies softmax to logits.

```python
def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)
```

### Display Images with Scores

This function displays images in a single row with corresponding scores.

```python
def display_images_with_scores(image_urls, scores, query_text):
    plt.figure(figsize=(20, 5))
    for i, (url, score) in enumerate(zip(image_urls, scores)):
        image = Image.open(requests.get(url, stream=True).raw)
        plt.subplot(1, len(image_urls), i + 1)
        plt.imshow(image)
        plt.title(f"Match: {score:.2%}")
        plt.axis('off')
    plt.suptitle(query_text, fontsize=16)
    plt.show()
```

## Tasks

### Task 1: Image Captioning

We generate a caption for a given image.

```python
image_url = "https://www.shutterstock.com/image-photo/scottish-gray-catsitting-on-entrance-600nw-1210863730.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
out = captioning_model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
display_image(image_url, title=caption)
```

### Task 2: Image-to-Text Retrieval

We find the best matching text for a given image from a text corpus.

```python
image_url = "https://www.shutterstock.com/image-photo/scottish-gray-catsitting-on-entrance-600nw-1210863730.jpg"
text_corpus = ["A cat playing in the park.", "A beautiful sunset over the ocean."]

image = Image.open(requests.get(image_url, stream=True).raw)
text_inputs = processor(text=text_corpus, return_tensors="pt", padding=True, truncation=True)
image_inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = retrieval_model(**image_inputs, **text_inputs)
    itm_scores = outputs.itm_score

print("ITM Scores:")
for score, text in zip(itm_scores, text_corpus):
    print(f"{text}: {score}")

best_idx = itm_scores[:, 1].argmax().item()
best_match = text_corpus[best_idx]

display_image(image_url, title=best_match)
```

### Task 3: Text-to-Image Retrieval

We find the best matching image for a given query text from a set of image URLs.

```python
query_text = "A beautiful sunset over the ocean."
image_urls = [
    "https://www.shutterstock.com/image-photo/scottish-gray-catsitting-on-entrance-600nw-1210863730.jpg",
    "https://images.fineartamerica.com/images-medium-large-5/1-beautiful-sunset-over-the-ocean-waters-ricardoreitmeyer.jpg",
    "https://wharfedaledoggyplaypark.co.uk/wp-content/uploads/2016/02/Wharfedale-doggy-play-park-happy-dogs.jpg"
]

text_inputs = processor(text=query_text, return_tensors="pt")
images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
image_inputs = processor(images=images, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = retrieval_model(**image_inputs, **text_inputs)
    itm_scores = outputs.itm_score

match_scores = torch.softmax(itm_scores, dim=1)[:, 1].cpu().numpy()
display_images_with_scores(image_urls, match_scores, query_text)
```

### Task 4: Visual Question Answering

We answer a question based on the content of an image.

```python
question = "What does the dog have in his mouth?"
image_url = "https://wharfedaledoggyplaypark.co.uk/wp-content/uploads/2016/02/Wharfedale-doggy-play-park-happy-dogs.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

inputs = processor(images=image, text=question, return_tensors="pt")

with torch.no_grad():
    outputs = vqa_model.generate(**inputs)
    answer = processor.decode(outputs[0], skip_special_tokens=True)

display_image(image_url, title=f"Q: {question}\nA: {answer}")
```

## Conclusion
This project showcases the versatility of the BLIP model in performing various language-image tasks, including image captioning, image-to-text retrieval, text-to-image retrieval, and visual question answering. The results demonstrate the model's capability to understand and generate coherent language descriptions of visual content. 
Further improvements can be made by fine-tuning the model on specific datasets relevant to the tasks.
