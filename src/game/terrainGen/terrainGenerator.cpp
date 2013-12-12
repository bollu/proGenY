#pragma once
#include "terrainGenerator.h"
#include "../../include/noise/PerlinNoise.h"
#include "../util/logObject.h"
#include "../util/mathUtil.h"

//form terrain taking noise values as a heightmap first
void genHeightShape(Terrain &terrain, PerlinNoise &noise) {
	unsigned int w = terrain.getWidth(), h = terrain.getHeight();

	unsigned int maxHeight = 0;

	int prevHeight2 = 0;
	int prevHeight1 = 0;

	for(unsigned int x = 0; x < w; x++) {
		float rawNoise = noise.noise((x * x * x)/ (float)(w * w * w), 0, 0);
		rawNoise *= 0.45;
		

		rawNoise = util::clamp<float>(rawNoise, 0, 1);

		float currentHeight = rawNoise * h;
		currentHeight = (currentHeight + prevHeight1 + prevHeight2) * 0.333;

		prevHeight2 = prevHeight1;
		prevHeight1 = currentHeight;

		if(std::floor(currentHeight) >= maxHeight) {
			maxHeight = currentHeight;
		}


		//fill up an entire column based on the heightmap
		for(int y = 0; y < std::floor(currentHeight); y++) {
			terrain.Set(x, y, terrainType::Filled);
		}

	}
	
	terrain.setMaxHeight(maxHeight);
}

void genBorders(Terrain &terrain) {
	unsigned int w = terrain.getWidth(), h = terrain.getHeight();

	for(int x = 0; x < w; x++) {
		terrain.Set(x, 0, terrainType::Filled);
		terrain.Set(x, h - 1, terrainType::Filled);
	}

	for(int y = 0; y < h; y++) {
		terrain.Set(0, y, terrainType::Filled);
		terrain.Set(w - 1, y, terrainType::Filled);
	}

}

void SetTerrain(Terrain &terrain, unsigned int x, unsigned int y, 
	unsigned int halfW, unsigned int halfH, terrainType value) {

	const unsigned int w = terrain.getWidth(), h = terrain.getHeight();

	for(int dy = -halfH; dy < halfH; dy++) {
		for(int dx = -halfW; dx < halfW; dx++) {
			int currentX = x + dx;
			int currentY = y + dy;

			currentX = util::clamp<int>(currentX, 0, w-1 );
			currentY = util::clamp<int>(currentY, 0, h-1 );

			terrain.Set(currentX, currentY, value);

		}
	}
}

void genCarver(Terrain &terrain, unsigned int seed, unsigned int steps, unsigned int thickness) {
	const unsigned int w = terrain.getWidth(), h = terrain.getHeight();

	unsigned int x = rand() % w;
	unsigned int y = terrain.getHeightAt(x);



	for (int i = 0; i < steps; i++) {
		
		SetTerrain(terrain, x, y, thickness / 2, thickness / 2, terrainType::Empty);
		terrain.Set(x, y, terrainType::Empty);

	
		for(int tries = 0; tries < 3; tries++) {
			int dy = rand() % 3 - 1;
			int dx = rand() % 3 - 1;

			int newX = util::clamp<int>(x + dx, 0, w-1 );
			int newY = util::clamp<int>(y + dy, 0, h-1 );
			if(terrain.At(newX, newY) == terrainType::Filled) {
				x = newX;
				y = newY;
			}
		}
	
	}
}

void genTerrain(Terrain &terrain, unsigned int seed){
	PerlinNoise noise(seed);

	genHeightShape(terrain, noise);

	int totalCarvers = vector2(terrain.getWidth(), terrain.getHeight()).Length();

	for(int i= 0; i < totalCarvers ; i++){
		genCarver(terrain, seed, totalCarvers / 4.0, 4);
	}

	genBorders(terrain);

	
	/*unsigned int w = terrain.getWidth(), h = terrain.getHeight();

	unsigned int maxHeight = 0;

	for(unsigned int y = 0; y < h; y++) {
		for(unsigned int x = 0; x < w; x++) {

			float rawNoise = noise.noise((x * x)/ (float)(w), (y * y)  / (float)(h), y);
			util::infoLog<<rawNoise<<util::flush;
			rawNoise -= 0.4;

			if(x == 0 || x == w - 1 || y == 0 || y == h - 1) {
				rawNoise = 1.0;
			}
			

			terrainType type = rawNoise <= 0 ? terrainType::Empty : terrainType::Filled;
			terrain.Set(x, y, type);

			if (y > maxHeight) {
				maxHeight = y;
			}
		}
	}

	terrain.setMaxHeight(maxHeight);
	*/
}


#include "../../core/Process/viewProcess.h"
vector2 getPlayerPosTerrain(Terrain &terrain, int terrainX){
	for(int delta = 0; ;delta++ ) {
		//start with y = 1 as y = 0 is solid ground
		for(int y = 1; y < terrain.getHeight(); y++) {
			if( terrain.At(terrainX + delta, y) == terrainType::Empty) {
				//y + 2 to clear the blocks completely (to the above the blocks basically)
				return vector2(terrainX + delta, y + 2); 

			}
		}
	}
};
