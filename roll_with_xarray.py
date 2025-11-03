import xarray as xr


def roll(img: xr.DataArray, dy=None, dx=None):
    img = img.copy()
    assert img.dims == ("y", "x"), img.dims
    if dy is None:
        dy = len(img.coords["y"]) // 2
    if dx is None:
        dx = len(img.coords["x"]) // 2
    img.coords["y"] = (img.coords["y"] + dy) % len(img.coords["y"]) - dy
    img.coords["x"] = (img.coords["x"] + dx) % len(img.coords["x"]) - dx
    return img.sortby(["y", "x"])
