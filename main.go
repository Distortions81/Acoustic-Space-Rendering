package main

import (
	"flag"
	"log"
	"os"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
)

// main configures the runtime, optionally records a profile, and launches Ebiten.
func main() {
	flag.Parse()
	boundaryReflect = *wallReflectFlag
	if boundaryReflect < 0 {
		boundaryReflect = 0
	} else if boundaryReflect > 1 {
		boundaryReflect = 1
	}
	viewportWidth = clampRange(*viewportWidthFlag, 1, w)
	viewportHeight = clampRange(*viewportHeightFlag, 1, h)
	blockWidth = clampRange(*blockWidthFlag, 1, w)
	blockHeight = clampRange(*blockHeightFlag, 1, h)
	var stopProfile func()
	if *recordDefaultPGO {
		var err error
		stopProfile, err = startDefaultPGORecording("default.pgo")
		if err != nil {
			log.Fatalf("unable to start PGO recording: %v", err)
		}
		defer stopProfile()
	}

	g := newGame()
	if *recordDefaultPGO {
		g.enableAutoWalk(pgoRecordDuration)
		go func(stop func()) {
			timer := time.NewTimer(pgoRecordDuration)
			<-timer.C
			stop()
			log.Printf("default.pgo captured after %s; exiting", pgoRecordDuration)
			os.Exit(0)
		}(stopProfile)
	}

	ebiten.SetWindowSize(displayWidth*windowScale, displayHeight*windowScale)
	ebiten.SetWindowTitle("Acoustic Steps")
	if err := ebiten.RunGame(g); err != nil {
		panic(err)
	}
}

func clampRange(value, minVal, maxVal int) int {
	if value < minVal {
		return minVal
	}
	if value > maxVal {
		return maxVal
	}
	return value
}
