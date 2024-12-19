### Introduction
This repository provides a **Grounded Segmentation Tool** that combines state-of-the-art object detection and segmentation models to identify and highlight objects in images based on user-defined keywords. By leveraging models such as **Grounding DINO** for object detection and **Segment Anything (SAM)** for mask generation, this tool offers a zero-shot approach to detecting and segmenting objects without requiring any additional training.

The tool is designed for flexibility and ease of use, allowing users to:

1. Specify an image (either a local file or a URL).
2. Provide a list of keywords in a text file to guide object detection.
3. Automatically generate segmentation masks for the detected objects.
4. Save the processed images with segmented regions for visualization or further analysis.
### Features
- **Zero-Shot Object Detection**: Detect objects in images without prior model fine-tuning.
- **Semantic Segmentation**: Generate precise masks for detected objects.
- **Keyword-Based Detection**: Use a list of keywords to customize object detection.
- **Image Enhancements**: Generate variations of images with altered brightness and contrast for improved visualization.
- **Polygon Refinement**: Convert masks into polygons for more accurate and interpretable segmentation.

This tool is ideal for applications in **computer vision, image annotation, object localization, and custom image analysis pipelines**. Whether you're a researcher, developer, or enthusiast, this tool provides a robust and straightforward way to extract meaningful insights from images.

### Usage
Run the script from the command line:

```
python segmentation_tool.py --image <image_path_or_url> --keywords <keywords_file> [--threshold <detection_threshold>] [--output <output_dir>]
```

##### Arguments:
- ```--image```: Path to the image file or a URL.
- ```--keywords```: Path to a text file containing the list of keywords (one per line).
- ```--threshold```: (Optional) Detection threshold. Default is 0.3.
- ```--output```: (Optional) Directory to save the output images. Default is ./output.
  
##### Example

```
python segmentation_tool.py \
    --image "./images/sample.jpg" \
    --keywords "./keywords.txt" \
    --threshold 0.3 \
    --output "./output"
```
###### Input
- Image File: A local image file or URL.
- Keywords File: A text file with one keyword per line.
Example ```keywords.txt```:

```
apple
banana
cat
```
###### Output

The output will be saved in the specified directory (./output by default) with each image labeled according to the detected objects.
