#pragma once
#include "gameSegment.h"



class Level{
public:

	void Startup();
	
	void Update();
	void Draw();
	
	void Shutdown();


	static void saveLevel(Level *level);
	static Level* loadLevel();
};
