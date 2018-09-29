#pragma once
int SoftenDiv = 25;
int Soften[] = { 
1, 1, 1, 1, 1,
1, 1, 1, 1, 1,
1, 1, 1, 1, 1,
1, 1, 1, 1, 1,
1, 1, 1, 1, 1 };

int SharpenDiv = 25;
int Sharpen[] = { 
-1, -1, -1, -1, -1,
-1, -1, -1, -1, -1,
-1, -1, 49, -1, -1,
-1, -1, -1, -1, -1,
-1, -1, -1, -1, -1 };

int DiagonalShatterDiv = 4;
int DiagonalShatter[] = { 
1, 0, 0, 0, 1,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
1, 0, 0, 0, 1 };

int HorizontalBlurDiv = 9;
int HorizontalBlur[] = { 
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
1, 2, 3, 2, 1,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0 };


// les filtres de sobel sont géré à part étant donné qu'il faut mettre en niveau de gris

int HorizontalSobelDiv = 1;
int HorizontalSobel[] = {
1, 2,  0, -2,  -1,
4, 8,  0, -8,  -4,
6, 12, 0, -12, -6,
4, 8,  0, -8,  -4,
1, 2,  0, -2,  -1 };

int VerticalSobelDiv = 1;
int VerticalSobel[] = { 
-1, -4, -6,  -4, -1,
-2, -8, -12, -8, -2,
0,  0,  0,   0,  0,
2,  8,  12,  8,  2,
1,  4,   6,  4,  1 };