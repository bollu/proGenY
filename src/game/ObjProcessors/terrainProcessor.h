#pragma once
#include "../../core/objectProcessor.h"
#include "../../include/SFML/Graphics.hpp"
#include "../../core/vector.h"
#include "../../core/Process/viewProcess.h"

class terrainProcessor : public objectProcessor{
private:
	b2World &world;
	sf::RenderWindow &window;
	viewProcess &view;

public:
	terrainProcessor(b2World &_world, sf::RenderWindow &_window, viewProcess &_view) : 
	world(_world), window(_window), view(_view) {

	};


	void Process(float dt){};
};