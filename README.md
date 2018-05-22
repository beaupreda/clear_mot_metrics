# CLEAR-MOT METRICS in Python

## Input data type

### XML

### SQLITE (Polytrack)
SQLite table named *bounding_boxes* organized as follow:
bounding_boxes
| --- |
object_id | frame_number | x_top_left | y_top_left | x_bottom_right | y_bottom_right
| --- | --- | --- | --- | --- | --- |
1 | 200 | 22 | 76 | 194 | 211 |
2 | 200 | 400 | 387 | 453 | 444 |
... |...|...|...|...|...|
*This is just an example*

## Dependencies
1. [Numpy](http://www.numpy.org/)
2. [Munkres](https://pypi.org/project/munkres/)

## Usage