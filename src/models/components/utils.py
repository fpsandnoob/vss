import json
import tifffile


def load_tiff_stack_with_metadata(file):
    """

    :param file: Path object describing the location of the file
    :return: a numpy array of the volume, a dict with the metadata
    """
    if not (file.name.endswith(".tif") or file.name.endswith(".tiff")):
        raise FileNotFoundError("File has to be tif.")
    with tifffile.TiffFile(file) as tif:
        data = tif.asarray()
        metadata = tif.pages[0].tags["ImageDescription"].value
    metadata = metadata.replace("'", '"')
    try:
        metadata = json.loads(metadata)
    except:
        print("The tiff file you try to open does not seem to have metadata attached.")
        metadata = None
    return data, metadata

def get_window(data, hu_max=256, hu_min=-150):
    # max_val = 3200
    # min_val = -2048
    
    max_val = 3096
    min_val = -1000
    
    data = data * (max_val - min_val) + min_val
    data = data.clip(hu_min, hu_max)
    data = (data - hu_min) / (hu_max - hu_min)
    return data
