# ReadMe

This project aims to solve sudoku while extracting the digits from an image .

## Approach

The approach is as follows:

1. Extract the puzzle area from the image and transform it into a square .
2. Extract cell of the puzzle.
3. Apply OCR to recognize the digits, if any append the recognised digit or else append 0 to the question string.
4. Now pass the string into the solver. The solver uses the algorithm proposed by [Peter Norvig](https://norvig.com/sudoku.html).


## Images

### Original Image
![Original Image](Img/original%20image.jpg)


### Finding the puzzle area in the image
![Contoured Image](Img/contoured%20image.jpg)


### Extarcting the Puzzle Area
![Cropped Image](Img/cropped%20image.jpg)


### Inverse Image of Puzzle Area
![Inverted Image](Img/inverted.jpg)


### Extracted Digits
![Extracted Digits](Img/Figure_1.png)
