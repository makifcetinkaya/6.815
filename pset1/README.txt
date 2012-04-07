LENGTH: 6.5 hours

Potential issues: I tested the functions I wrote without the given test file.
But there might be some index bound issues with the arrays. I guess I could have
written better error check mechanism for this, but I didn't want to spend too
much time on the code, but rather understand the basics of image properties.

Collaboration: For calculation of (luminance, chrominance) pair I was not sure
which [R, G, B] values to pass to BW(im, weights=<defaults>) function. On Piazza Daniel
Sngiem told me that the <defaults> are to be used for black-white conversion.

Unclear/Difficult: The values to pass to BW to get luminance of image was not clear.
As I read on wikipedia, calculation of luminance differ a lot. But it would be nice
to have a footnote about this so that I didn't spend 1.5 hour trying to figure this out.
Well, maybe my silliness, too; I tried every RGB combination, just not the defaults.

Most exciting: YUV-coded images. I tested rgb2yuv function I wrote myself, and checked 
out the yuv-saved image. The image was just like the TV-image when especially tube-TV s
have some antenna problem.