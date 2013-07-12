#pragma once
#include "gunDataGenerator.h"
#include "bulletDataGenerator.h"


gunDataGenerator::gunDataGenerator(unsigned int power, unsigned long seed){
	this->power = power;
	this->seed = seed;

	generator.seed(seed);
};

gunData gunDataGenerator::Generate(){
	unsigned int clipSize;
	unsigned int clipCooldown;
	unsigned int shotCooldown;
	float bulletRadius;
	float bulletVel;
	bulletData bullet;

	bulletDataGenerator bulletDataGen(this->power, this->seed);

	clipSize = 3 + (generator() % 5);
	clipCooldown = 10 + (generator() % 10);
	shotCooldown = 0 + (generator() % 5);
	bulletRadius = 0.5 + ((generator() % 3) / 3.0);
	bulletVel = 10 + (generator() % 40);
	bullet = bulletDataGen.Generate();

	gunData data;
	data.setClipSize(clipSize);
	data.setClipCooldown(clipCooldown);
	data.setShotCooldown(shotCooldown);
	data.setBulletRadius(bulletRadius);
	data.setBulletVel(bulletVel);
	data.setBulletData(bullet);
};

