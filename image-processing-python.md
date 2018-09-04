# python scripts for image processing

## Color segmentation
### kmeans
```python
import os
import glob
from tqdm import tqdm
from skimage import io
from sklearn.cluster import KMeans

def kmeans_color_clustering(inputs_dir, K):
    # get images
    inputs_fns = glob.glob(os.path.join(inputs_dir, '*.png'))
    for inputs_fn in tqdm(inputs_fns):
        fn_only = os.path.splitext(os.path.basename(inputs_fn))[0]

        # read as RGB
        inputs_image = io.imread(inputs_fn)
        inputs_image = inputs_image[:, :, :3]

        # prepare to cluster(segment) color image
        rows, cols, bands = inputs_image.shape
        total_pixels = rows * cols
        x = inputs_image.reshape(total_pixels, bands)

        kmeans = KMeans(n_clusters=top_n, random_state=0).fit(x)
        labels = kmeans.labels_.reshape(rows, cols)

        canvas = np.ones_like(inputs_image, dtype=np.uint8) * 255
        for i in np.unique(labels):
            # get color
            color = np.around(kmeans.cluster_centers_[i])
            color = color.astype(np.uint8)

            # location where should be colored as current color
            indices = labels == i

            # set color
            canvas[indices] = color

        output_fn = os.path.join(output_dir, '{:s}.png'.format(fn_only))
        canvas = cv2.cvtColor(canvas, code=cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_fn, canvas)

    return
``` 