#pragma once
#include "worldProcess.h"
#include "../componentSys/Object.h"


class worldAABBQuery : public b2QueryCallback{
private:
	std::vector<Object *> object;
public:
	
	bool ReportFixture(b2Fixture* fixture);
};
