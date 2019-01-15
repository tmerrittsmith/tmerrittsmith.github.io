#### Back with a bang!

Well, if you're reading this then you're interested in the beginning of the story. 

Having worked as a teacher for 7 years, I finally made the move in to data science in 2017. Naturally, the next step was to revitalise the old \*.github.io and post up all of those awesome... post up one or two things that I managed to scrape together. It took another 18 months, but here we are.

Since it took me over an hour to serve up the 2016 jekyll site, it seems appropriate to write down what I did, on my new Ubuntu 18.04.1 LTS. These steps should help once you've cloned down that old repo you've had lying around for too long.

- Edit the `Gemfile.lock`. Right at the bottom, make sure that `BUNDLED-WITH` has the correct version (you can check it with `bundle version`)
- I'm basically a linux noob, so I also had problems until I upgraded all my packages, and installed build tools (h/t computingforgeeks):
`sudo apt upgrade
sudo apt install make build-essential`
- I also had to `sudo apt install zlib1g-dev`
- If you're on a new ubuntu installation, you'll need to install jekyll, so cd into your pages directory and run `bundle install`
- Then you can run `bundle update` to make sure all the gems are... *sparkly and up to date*

If you're good to go, then throw out a `bundle exec jekyll serve` and that's it.
Back. In. The. Game.











Game on. 
