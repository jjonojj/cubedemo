use std::collections::HashMap;

use sdl3::keyboard::Keycode;

pub struct Keymap {
    states: HashMap<Keycode, bool>,
}

impl Keymap {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
        }
    }

    pub fn press(&mut self, kc: Keycode) {
        self.states.insert(kc, true);
    }

    pub fn release(&mut self, kc: Keycode) {
        self.states.insert(kc, false);
    }

    pub fn get(&self, kc: Keycode) -> bool {
        *self.states.get(&kc).unwrap_or(&false)
    }
}
