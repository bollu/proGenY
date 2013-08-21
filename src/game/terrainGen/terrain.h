#pragma once
#include "../../core/vector.h"
#include "tile.h"
#include "room.h"
#include <vector>

namespace terrainGen{
	class Terrain{
		std::vector<Tile> tiles;
		vector2 dim;
	public:
		Terrain(vector2 dim);
		
		Tile &getTile(vector2 pos);
		vector2 getDim();
	};
};