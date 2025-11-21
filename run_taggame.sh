#!/bin/bash

# Check if maven is installed
if ! command -v mvn &> /dev/null; then
    echo "Installing maven..."
    sudo apt install maven -y
    echo "Maven installed successfully!"
fi

# Navigate to taggame-java, compile, and run
cd taggame-java && mvn compile && java -cp "target/classes:lib/geom2D/javaGeom-0.11.1.jar:lib/slick2D/slick.jar:lib/slick2D/lwjgl.jar:lib/slick2D/lwjgl_util.jar:$HOME/.m2/repository/org/json/json/20231013/json-20231013.jar" -Djava.library.path=natives taggame.SlickGraphicsRunner
