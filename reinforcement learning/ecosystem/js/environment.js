var Environment = function(two) {

	this.two = two;

	this.width = two.width;
	this.height = two.height;

	num_red = 2;
	num_green = 8;

	this.balls = []
	this.green_balls = []
	this.red_balls = []

	// Create the predators
	let id = 0;
	for (i = 0; i < num_red; i++) {
		let x = 0; // make random
		let y = 0; // make random
		let ball = new Ball(x, y, 'red', id, this);
		this.balls.push(ball);
		this.red_balls.push(ball);
		id++;
	}

	// Create the prey
	for (i = 0; i < num_green; i++) {
		let x = 0; // make random
		let y = 0; // make random
		let ball = new Ball(x, y, 'green', id, this);
		this.balls.push(ball);
		this.green_balls.push(ball);
		id++;
	}

	let water_x = this.width/2
	let water_y = this.width/2
	this.water = new Ball(this.width/2, this.height/2, 'blue', num_red + num_green, this)
	this.balls.push(this.water);

	this.state_dim = (num_red + num_green + 1) * 4; // x, y, dx, dy for red, green, water

	this.touching = function(ball1, ball2) {
		let distance = Math.sqrt(Math.pow(ball1.x - ball2.x, 2) + Math.pow(ball1.y - ball2.y, 2))
		let radii = ball1.radius + ball2.radius;
		if (distance < radii) {
			return true;
		} else {
			return false;	
		}
	}

	// Given the id of the ball, return the vector representing its state
	this.perception = function(ball_id) {
		let self_perception = [];
		let others_perception = [];

		for (i = 0; i < len(this.balls); i++) {
			let ball = this.balls[i];

			// If the ball is the perceiving agent, (ball.id == ball_id)
			if (ball.id == ball_id) {
				self_perception.push(ball.x);
				self_perception.push(ball.y);
				self_perception.push(ball.dx);
				self_perception.push(ball.dy);
			} else {
				others_perception.push(ball.x);
				others_perception.push(ball.y);
				others_perception.push(ball.dx);
				others_perception.push(ball.dy);
			}
		}

		others_perception.push(this.water.x);
		others_perception.push(this.water.y);
		others_perception.push(this.water.dx);
		others_perception.push(this.water.dy);
		
		return self_perception.concat(others_perception);
	}

	this.reward = function(ball_id) {
		let reward = 0.0;
		let ball = this.balls[ball_id];
		let water = this.balls[(this.balls.length-1)]

		let red_predator_reward = 2.0;
		let green_prey_reward = -10.0;
		let green_water_reward = 1.0;

		if (ball.color == 'red') {
			for (i = 0; i < this.green_balls.length; i++) {
				if this.touching(ball, this.green_balls[i]) {
					reward += red_predator_reward;
				}
			}
		} else {
			for (i = 0; i < this.red_balls.length; i++) {
				if this.touching(ball, this.red_balls[i]) {
					reward += green_prey_reward;
				}
				if this.touching(ball, this.water) {
					reward += green_water_reward;
				}
			}
		}
	}

	this.draw = function() {
		// clear background 


		// draw balls
		for (i = 0; i < this.balls.length; i++) {
			let ball = this.balls[i];
			ball.draw();
		}
	}

	this.update = function() {
		for (i = 0; i < this.balls.length; i++) {
			let ball = this.balls[i];
			ball.update();
		}
		this.draw();
	}
}

module.exports = Environment;