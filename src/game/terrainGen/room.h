#pragma once
#include "../../core/vector.h"
#include "tile.h"
#include "tileChunk.h"
#include "terrain.h"

namespace terrainGen{
	
	class Room : public tileChunk{

	public:
		static Room Center(Terrain &terrain, vector2 center, vector2 halfDim);
		static Room Extremes(Terrain &terrain, vector2 bottomLeft, vector2 topRight);
		static Room entireTerrain(Terrain &terrain);
		
		vector2 getCenter();
		vector2 getHalfDim();
		vector2 getBottomLeft();
		vector2 getTopRight();

		unsigned int getNumTiles();

		void walkTiles();

	private:
		vector2 bottomLeft;
		vector2 topRight;

		vector2 halfDim;
		vector2 center;

		unsigned int numTiles;
		Terrain &terrain;

		Room(Terrain &terrain, vector2 center, vector2 halfDim);


	};
};