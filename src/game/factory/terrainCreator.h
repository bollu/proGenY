#pragma once
#include "objectFactory.h"
#include "../../core/Process/viewProcess.h"
#include "../terrainGen/terrainGenerator.h"



class terrainCreator : public objectCreator{
public:
	terrainCreator(viewProcess *viewProc);

	void setBounds(vector2 bottomLeft, vector2 topRight, vector2 chunkDim);
	void getEmptySpace();

	Object *createObject();
private:

	typedef int chunkType;
	struct Chunk{
		bool filled;
	};

	vector2 bottomLeft, topRight;

	unsigned int totalChunkCount;
	vector2 numChunks;

	std::vector<Chunk> chunks;

	viewProcess *viewProc;


	//quantized coord- [(0, 0) to (numChunks.x - 1, numChunks.y - 1)] 
	vector2 _game2QuantizedCoord(vector2 realCoord);
	vector2 _quantized2WorldCoord(vector2 quantizedCoord);
	vector2 _limitQuantizedCoord(vector2 rawQuantizedCoord);
	
	//position in terrain's coordinate system
	Chunk& _getChunk(vector2 quantizedPos);

	//flood fills chunks in the bounding rectangle.center of rectangle
	//is quantizecdCenter. Overwrites values. 
	int _floodFillChunks(vector2 quantizedCenter, vector2 halfDim, chunkType type);
	int _negateChunks(vector2 quantizedCenter, vector2 halfDim);

	float _getDensity();
	
	void _genTerrainData();
	void _dbgPrintChunks();

};