#pragma once
#include "terrain.h"
#include <iostream>

using namespace terrainGen;

Terrain::Terrain(vector2 dim){
	this->dim = dim;
	assert(dim.x > 0 && dim.y > 0);

	PRINTVECTOR2(dim)

	this->tiles.reserve(dim.x * dim.y);
		
	vector2 pos = vector2(0, 0);
	for(pos.y = 0; pos.y < dim.y; pos.y++){
		for(pos.x = 0; pos.x < dim.x; pos.x++){
			this->tiles.push_back(Tile(pos, 0, false));
		}
	}	
}

Tile& Terrain::getTile(vector2 pos){
	assert(pos.x >= 0 && pos.y >= 0);
	assert(pos.x < dim.x && pos.y < dim.y);
	Tile &t =  this->tiles[(int)(pos.x + dim.y * pos.y)];
	return t;
};

vector2 terrainGen::Terrain::getDim(){
	return this->dim;
};
