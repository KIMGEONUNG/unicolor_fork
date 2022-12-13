## Todo list

- Understand the process from user input hint representation to model condition representation
- Implement a feature converting the gradio maks to model condition.
- Check the model code is available
- Construct Gradio GUI only for stroke-based approach, which is the most fundamental way
  - Please focus on it....

## Done

- Check the training code is available --> not yet


## Code Bookmark


### what does index mean?

### Location of the model codes 

Check the below lists
- unicolor/framework/hybrid\_tran/models/transformer.py
- unicolor/framework/hybrid\_tran/models/vqgan.py
- unicolor/framework/chroma\_vqgan/models/modules.py

### Unicolor use LAB fusion

```python
# line 101 in utils_func.py 
def draw_color(l, color, rect):
    y0, y1, x0, x1 = rect
    l = np.array(l.convert('RGB'))
    lab = cv2.cvtColor(l, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0:1]  
    ab = lab[:, :, 1:3] 
    draw = np.array(color).astype(np.uint8)
    if len(draw.shape) == 1:
        draw = np.expand_dims(draw, axis=[0, 1])
    draw = cv2.cvtColor(draw, cv2.COLOR_RGB2LAB)
    ab[y0:y1, x0:x1, :] = draw[:, :, 1:3]  # extract chrominance
    lab = np.concatenate([l, ab], axis=2)  # Concat luma and chroma
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img)
``` 

