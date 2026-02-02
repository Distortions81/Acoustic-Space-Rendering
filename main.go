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
	if worldBoundaryAbsorbFlag != nil && *worldBoundaryAbsorbFlag {
		boundaryReflect = 0
	} else {
		boundaryReflect = *worldBoundaryReflectFlag
	}
	if boundaryReflect < 0 {
		boundaryReflect = 0
	} else if boundaryReflect > 1 {
		boundaryReflect = 1
	}

	roomWallSegments = defaultWallSegments
	if roomWallSegmentsFlag != nil {
		roomWallSegments = *roomWallSegmentsFlag
	}
	roomWallMinLenM = defaultWallMinLenM
	if roomWallMinLenMFlag != nil {
		roomWallMinLenM = *roomWallMinLenMFlag
	}
	roomWallMaxLenM = defaultWallMaxLenM
	if roomWallMaxLenMFlag != nil {
		roomWallMaxLenM = *roomWallMaxLenMFlag
	}
	roomWallThicknessM = defaultWallThicknessM
	if roomWallThicknessMFlag != nil {
		roomWallThicknessM = *roomWallThicknessMFlag
	}
	roomWallThicknessJitM = defaultWallThicknessJitM
	if roomWallThicknessJitterMFlag != nil {
		roomWallThicknessJitM = *roomWallThicknessJitterMFlag
	}
	roomWallExclusionM = defaultWallExclusionM
	if roomWallExclusionRadiusMFlag != nil {
		roomWallExclusionM = *roomWallExclusionRadiusMFlag
	}
	roomWallReflect = defaultRoomWallReflect
	if roomWallReflectFlag != nil {
		roomWallReflect = *roomWallReflectFlag
	}
	if roomWallReflect < 0 {
		roomWallReflect = 0
	} else if roomWallReflect > 1 {
		roomWallReflect = 1
	}
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

	ebiten.SetWindowSize(w*windowScale, h*windowScale)
	ebiten.SetWindowTitle("Acoustic Steps")
	if err := ebiten.RunGame(g); err != nil {
		panic(err)
	}
}
