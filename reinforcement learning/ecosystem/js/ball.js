var Ball = function(x, y, color, id, environment) {

	this.environment = environment;
	this.id = id;
	this.color = color;
	if (color == 'red') {
		this.radius = 5.0;
	} else if (color == 'blue') {
		this.radius = 3.0;
	} else {
		this.radius = 1.0
	}


	// Movement

	this.x = x;
	this.y = y;
	this.dx = 0.0;
	this.dy = 0.0;
	this.top_speed = 5.0

	this.go_left = function() {
		if this.dx > -this.top_speed {
			this.dx -= 1.0;
		}
	}

	this.go_right = function() {
		if this.dx < this.top_speed {
			this.dx += 1.0;
		}
	}

	this.go_up = function() {
		if this.dy > -this.top_speed {
			this.dy -= 1.0;
		}
	}

	this.go_down = function() {
		if this.dy < this.top_speed {
			this.dy += 1.0;
		}
	}

	this.move = function() {
		
		if (this.x > this.environment.width) {
			this.x = this.environment.width;
			this.dx = 0;
		} else if (this.x < 0) {
			this.x = 0;
			this.dx = 0;
		}
		if (this.y > this.environment.height) {
			this.y = this.environment.height;
			this.dy = 0;
		} else if (this.y < 0) {
			this.y = 0;
			this.dy = 0;
		}

		this.x += this.dx;
		this.y += this.dy;
	}


	// Learning and action

	let env = {};
	env.getNumStates = function() { return this.environment.state_dim; }
	env.getMaxNumActions = function() { return 4; }
	let spec = {};
	this.brain = new RL.DQNAgent(env, spec); 

	this.do = function(action) {
		if (action == 0) {
			this.go_left();
		} else if (action == 1) {
			this.go_right();
		} else if (action == 2) {
			this.go_up();
		} else if (action == 3) {
			this.go_down();
		}
	}

	this.decide = function() {
		state = this.environment.perception(this.id);
		return this.brain.act(state);
	}

	this.learn = function() {
		let reward = this.environment.reward(this.id);
		this.brain.learn(reward);
	}


	// Drawing

	this.draw = function() {
		let  = this.environment.d3;

	}


	// Loop function

	this.update = function(action) {
		this.do(this.decide());
		this.move();
		this.learn();
	}
}

module.exports = Ball;