#pragma once

#include "globals.h"


void processNormalKeys(unsigned char key, int x, int y) {
	switch (key) {
	case '+':
		DELTA = max(DELTA - 50.0, 50.0);
		break;
	case '-':
		DELTA += 50;
		break;
	case ' ':
		PLAY = !PLAY;
	}
	UPDATE_GLOBAL(DELTA, float);
}


void processSpecialKeys(int key, int x, int y) {
	switch (key) {
	case GLUT_KEY_RIGHT:
		X_CENTER += 50;
		break;
	case GLUT_KEY_LEFT:
		X_CENTER -= 50;
		break;
	case GLUT_KEY_UP:
		Y_CENTER += 50;
		break;
	case GLUT_KEY_DOWN:
		Y_CENTER -= 50;
		break;
	}
	UPDATE_GLOBAL(X_CENTER, float);
	UPDATE_GLOBAL(Y_CENTER, float);
}
