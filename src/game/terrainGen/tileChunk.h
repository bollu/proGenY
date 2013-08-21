#pragma once
#include "tile.h"





namespace terrainGen{
	class Terrain;
	class tileChunkVisitor;

	class tileChunk{
	public:
		virtual ~tileChunk(){};
		virtual unsigned int getNumTiles() = 0;
//		virtual void Walk(tileChunkVisitor &v) = 0;
	};

	class tileChunkVisitor{
		virtual void visitTile(Tile &t) = 0;	
	};

};

