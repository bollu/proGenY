#include "terrainGenerator.h"
#include "../../include/noise/tinymt32.h"

#include "../../core/IO/logObject.h"
#include "../../core/math/mathUtil.h"

#include <deque>


#define float01 tinymt32_generate_float01


void genBorders(Terrain &terrain) {
	const unsigned int w = terrain.getWidth(), h = terrain.getHeight();

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

	for(int dy = -halfH; dy <= halfH; dy++) {
		for(int dx = -halfW; dx <= halfW; dx++) {
			int currentX = x + dx;
			int currentY = y + dy;

			currentX = util::clamp<int>(currentX, 0, w-1 );
			currentY = util::clamp<int>(currentY, 0, h-1 );

			terrain.Set(currentX, currentY, value);

		}
	}
}


void genAreas(Terrain &terrain, tinymt32_t &mt, vector2 minRoomSize){
	


}

void genTerrain(Terrain &terrain, unsigned int seed){
	const unsigned int w = terrain.getWidth(), h = terrain.getHeight();

	tinymt32_t mt;
	tinymt32_init(&mt, seed);

	
	genBorders(terrain);

	for(int i = 0; i < w; i++) {
		for(int j = 0; j < h; j++) {
			if(rand() % 5 == 0){
				terrain.Set(i, j, terrainType::Filled);
			}
		}
	}
	
}


#include "../../core/Rendering/viewProcess.h"
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


bool shouldSubdivide(Terrain &terrain, int currentX, int h){
	const unsigned int w = terrain.getWidth();

	for(int x = currentX; x < w; x++) {

		//if it's filled, then you have to subdivide.
		if(terrain.At(x, h) == terrainType::Filled){
			return true;
		}
	}

	return false;
};


//this assumes that subdivision exists. it modifies currentX to the latest value.
AABB Subdivide_(Terrain &terrain, int &currentX, int h){
	const unsigned int w = terrain.getWidth();

	while(terrain.At(currentX, h) == terrainType::Empty) {
		currentX++;
	}

	vector2 begin = vector2(currentX, h);

	//keep going till you hit a block whose neighbor is empty
	while((currentX + 1) < w && terrain.At(currentX + 1, h) == terrainType::Filled) {
		currentX++;
	}

	vector2 end = vector2(currentX + 1, h + 1);

	//step currentX one more time to escape the filled block
	currentX++;

	return AABB::Endpoints(begin, end);
};

std::vector<AABB> genTerrainChunks(Terrain &terrain) {
	const unsigned int w = terrain.getWidth(), h = terrain.getHeight();

	std::vector<AABB> AABBarr;

	for(unsigned int y = 0; y < h; y++) {
		int currentX = 0;

		while(currentX < w) {
			if(shouldSubdivide(terrain, currentX, y)){
				AABBarr.push_back(Subdivide_(terrain, currentX, y));
			}
		}

	}

	return AABBarr;
};
