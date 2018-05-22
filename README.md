# CLEAR-MOT METRICS in Python
Implementation of the multiple object tracking metrics as defined in this article:

Keni, Bernardin, and Stiefelhagen Rainer. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." EURASIP Journal on Image and Video Processing 2008 (2008)

[article](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.367.6279&rep=rep1&type=pdf)

## Input data type
### XML
XML file as the example below:

```xml
<?xml version="1.0" ?>
<Video end_frame="400" fname="/path/to/filename.jpg" start_frame="1">
	<Trajectory end_frame="400" obj_id="0" obj_type="car" start_frame="1">
		<Frame annotation="0" contour_pt="0" frame_no="1" height="91" observation="0" width="141" x="489" y="323"/>
		<Frame annotation="0" contour_pt="0" frame_no="2" height="95" observation="0" width="146" x="493" y="323"/>
		<Frame annotation="0" contour_pt="0" frame_no="3" height="96" observation="0" width="147" x="498" y="326"/>
	</Trajectory>
	<Trajectory end_frame="400" obj_id="4" obj_type="car" start_frame="1">
		<Frame annotation="0" contour_pt="0" frame_no="1" height="37" observation="0" width="40" x="272" y="195"/>
		<Frame annotation="0" contour_pt="0" frame_no="2" height="41" observation="0" width="23" x="125" y="303"/>
		<Frame annotation="0" contour_pt="0" frame_no="3" height="33" observation="0" width="23" x="623" y="197"/>
	</Trajectory>
	...
</Video>
```
Only some attributes are necessary for each tag:

#### Video
* end_frame
* start_frame

#### Trajectory
* obj_id

#### Frame
* frame_no
* width
* height
* x
* y

### SQLITE (Polytrack)
SQLite table named *bounding_boxes* organized as follow:

| object_id | frame_number | x_top_left | y_top_left | x_bottom_right | y_bottom_right |
| --------- | ------------ | ---------- | ---------- | -------------- | -------------- |
| 1         | 200          | 22         | 76         | 194            | 211            |
| 2         | 200          | 400        | 387        | 453            | 444            |
| ...       | ...          | ...        | ...        | ...            | ...            |

## Dependencies
1. [Python 3](https://www.python.org/download/releases/3.0/)
2. [Munkres](https://pypi.org/project/munkres/)
3. [Numpy](http://www.numpy.org/)

## Usage
```
python3 MOT_Metrics.py path/to/annotations/file path/to/hypotheses/file evaluation_method distance
```
evaluation_method is either:
* bboverlap
* centroid
e.g.
```
python3 MOT_Metrics.py ./rene_gt.sqlite ./rene_vibe.sqlite bboverlap 0.5
```

## License
See the LICENSE file for more details.