#pragma once

//category = I am a ...
//mask = I collide with...
enum CollisionGroups {
	BULLET = 2,
	ENEMY = 4,
	PLAYER = 8,
	PICKUP = 16,
	TERRAIN = 32,
};