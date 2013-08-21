#pragma once
#include <vector>
#include "../../core/vector.h"


namespace terrainGen
{
    class terrainGenerator
    {
private:
public:
	    virtual 
	    void GenerateTerrrain () = 0;

	    virtual ~terrainGenerator ();
    };
}