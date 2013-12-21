#include "GunDataGenerator.h"
#include "Generator.h"

void _genBulletData(GunData &gunData, BulletGenData &generationData){

	BulletData bulletData = GenBulletData(generationData);

	bulletData.addEnemyCollision(Hash::getHash("enemy"));
	bulletData.addEnemyCollision(Hash::getHash("dummy"));

	bulletData.addIgnoreCollision(Hash::getHash("bullet"));
	bulletData.addIgnoreCollision(Hash::getHash("player"));
	
	gunData.setBulletData(bulletData);

};


void _genRocket(std::mt19937 &generator, unsigned long seed, unsigned int power, GunData &gunData){
	/* quite a low rate of fire, average
	clip size, heavy damage
	*/

	//clip size can vary from power-6  to powe6   
	gunData.setClipSize(1);
	gunData.setShotCooldown(0);


	float clipCD = normGenInt(generator, 16 + power, 2) * 0.01;
	gunData.setClipCooldown(clipCD);

	gunData.setBulletRadius(genFloat(generator, 0.8, 0.9));
	gunData.setBulletVel(genFloat(generator, 40, 60));

	BulletGenData bulletGenData;
	bulletGenData.seed = seed;
	bulletGenData.power = power;
	bulletGenData.numAbilities = 1;
	bulletGenData.abilityPower = 1;
	bulletGenData.gravity = gravityProperty::highGravity;
	bulletGenData.damage = damageProperty::lowDamage;
	bulletGenData.knockback = knockbackProperty::noKnockback;

	_genBulletData(gunData, bulletGenData);
};


void _genMachineGun(std::mt19937 &generator, unsigned long seed, unsigned int power, GunData &gunData){

	/*high rate of fire, small bullets, medium clip size */

	gunData.setClipSize(normGenInt(generator, 20, 2));

	int shotCooldown = normGenInt(generator, 3, 1);
	gunData.setShotCooldown(shotCooldown);
	gunData.setClipCooldown(normGenInt(generator, shotCooldown + 20, 2));
	gunData.setBulletRadius(genFloat(generator, 0.4, 0.5));
	gunData.setBulletVel(genFloat(generator, 20, 30));

	BulletGenData bulletGenData;
	bulletGenData.seed = seed;
	bulletGenData.power = power;
	bulletGenData.numAbilities = 1;
	bulletGenData.abilityPower = 1;
	bulletGenData.abilityPower = 1;
	bulletGenData.gravity = gravityProperty::lowGravity;
	bulletGenData.damage = damageProperty::lowDamage;
	bulletGenData.knockback = knockbackProperty::noKnockback;

	_genBulletData(gunData, bulletGenData);
};


GunData GenGunData(GunGenData &genData){
	GunData gunData;
	static std::mt19937 generator;

	generator.seed(genData.seed);

	switch(genData.type) {
		case GunType::machineGun:
			_genMachineGun(generator, genData.seed, genData.power, gunData);
			break;
		
		case GunType::Rocket:
			_genRocket(generator, genData.seed, genData.power, gunData);
			break;
	}

	return gunData;
};