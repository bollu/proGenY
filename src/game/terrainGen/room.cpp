#pragma once
#include "room.h"

using namespace terrainGen;


Room::Room(Terrain &_terrain, vector2 center, vector2 halfDim) : terrain(_terrain){
	this->center = center;
	this->halfDim = halfDim;

	this->bottomLeft =  center - halfDim;
	this->topRight = center + halfDim; 

	this->numTiles = (topRight.x - bottomLeft.x) *
	(topRight.y - bottomLeft.y);
}



Room Room::Center(Terrain &terrain, vector2 center, vector2 halfDim) {
	return Room(terrain, center, halfDim);
}

Room Room::Extremes(Terrain &terrain, vector2 bottomLeft, vector2 topRight) {
	vector2 center = (bottomLeft + topRight) * 0.5;
	vector2 halfDim = topRight - center;

	return Room(terrain, center, halfDim);
}

Room Room::entireTerrain(Terrain &terrain){
	vector2 center = terrain.getDim() * 0.5;
	return Room(terrain, center, center);
};


vector2 Room::getCenter(){
	return this->center;
}

vector2 Room::getHalfDim(){
	return this->halfDim;
}

vector2 Room::getBottomLeft(){
	return this->bottomLeft;
}

vector2 Room::getTopRight(){
	return this->topRight;
}

unsigned int Room::getNumTiles(){
	return this->numTiles;
}

void Room::walkTiles(){

};