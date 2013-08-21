#pragma once
#include "objectFactory.h"
#include "../../core/Process/viewProcess.h"
#include "../terrainGen/terrain.h"

/*\sa responsible for creating the terrain object
   all interaction with the object is generally done in
   "chunk" coordinates. Chunks are similar to tiles. Once
   the terrain creator is created, all interaction with this object
   in terms of coordinate systems will be done in terms of chunks only
 */
class terrainCreator : public objectCreator
{
public:
	terrainCreator ( viewProcess *viewProc ); 
	void setBounds ( vector2 bottomLeft, vector2 topRight, vector2 chunkDim );
	void reserveRectSpace ( vector2 center, vector2 halfDim );

	Object *createObject ();


private:
//	terrainGenerator terrainGen;
	vector2 bottomLeft, topRight, numChunks;
	int totalChunkCount;
	viewProcess *viewProc;

	void _dbgPrintChunks ();
	void _genTerrainData ();
};