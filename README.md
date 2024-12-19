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

####### Input
- Image File: A local image file or URL.
- Keywords File: A text file with one keyword per line.
Example ```keywords.txt```:

```
apple
banana
cat
```
####### Output

The output will be saved in the specified directory (./output by default) with each image labeled according to the detected objects.
