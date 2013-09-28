#pragma once
#pragma once
#include "objectFactory.h"
#include "../../core/Rendering/viewProcess.h"
#include "../ObjProcessors/offsetProcessor.h"

#include "../../core/componentSys/processor/renderProcessor.h"
#include "../../core/componentSys/processor/phyProcessor.h"
#include "../../core/Rendering/renderUtil.h"

#include "../defines/renderingLayers.h"


class terrainCreator : public objectCreator{
public:
	terrainCreator(viewProcess *_viewProc);
	Object *createObject(vector2 bottomLeft) const;
	
	void Init(vector2 numBlocks, vector2 blockDim, unsigned int seed);

private:
	void _createFilledBlock(phyData &phy, vector2 blockPos) const;
	renderData _createRenderer(phyData &phy) const;
	
	viewProcess *viewProc;
	vector2 numBlocks;
	vector2 blockDim;
	unsigned int seed;

	vector2 _block2World(vector2 blockPos) const;

};
