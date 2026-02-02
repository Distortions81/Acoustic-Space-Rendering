package main

import (
	"flag"
	"log"
	"os"
	"strings"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
)

// main configures the runtime, optionally records a profile, and launches Ebiten.
func main() {
	flag.Parse()
	set := map[string]bool{}
	flag.CommandLine.Visit(func(f *flag.Flag) {
		set[f.Name] = true
	})
	if worldBoundaryAbsorbFlag != nil && *worldBoundaryAbsorbFlag {
		worldBoundaryReflect = 0
	} else {
		worldBoundaryReflect = *worldBoundaryReflectFlag
	}
	if worldBoundaryReflect < 0 {
		worldBoundaryReflect = 0
	} else if worldBoundaryReflect > 1 {
		worldBoundaryReflect = 1
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
	material := defaultRoomWallMaterial
	if roomWallMaterialFlag != nil {
		material = strings.ToLower(strings.TrimSpace(*roomWallMaterialFlag))
	}
	if coeff, ok := roomWallMaterialReflectivity(material); ok {
		roomWallReflect = coeff
	}
	// Manual numeric override takes precedence when explicitly provided.
	if set["room-wall-reflect"] && roomWallReflectFlag != nil {
		roomWallReflect = *roomWallReflectFlag
	}
	if roomWallReflect < 0 {
		roomWallReflect = 0
	} else if roomWallReflect > 1 {
		roomWallReflect = 1
	}

	if rt60AutoFlag != nil && *rt60AutoFlag && !set["rt60-s"] && rt60SecondsFlag != nil {
		worldWidthM := cellSizeMeters() * float64(w)
		*rt60SecondsFlag = estimateRT60Seconds(worldWidthM, material, roomWallReflect)
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

func roomWallMaterialReflectivity(material string) (float64, bool) {
	switch material {
	case "", "drywall", "plaster":
		return 0.90, true
	case "concrete":
		return 0.95, true
	case "brick":
		return 0.93, true
	case "glass":
		return 0.96, true
	case "wood", "panel", "paneling":
		return 0.88, true
	case "curtain", "drape":
		return 0.70, true
	case "acoustic", "foam", "tile":
		return 0.45, true
	default:
		return 0, false
	}
}

func estimateRT60Seconds(worldWidthM float64, roomWallMaterial string, roomWallReflect float64) float64 {
	// Heuristic for this prototype:
	// - RT60 tends to scale roughly linearly with room size.
	// - 2D wave simulation rings more than real 3D rooms, so we bias shorter.
	// - Material presets nudge the result, but this is not a full Sabine/Eyring model.
	//
	// "worldWidthM" is used as a proxy for the room's characteristic length.
	L := worldWidthM
	if L <= 0 {
		L = float64(defaultWorldWidthFeet) * 0.3048
	}
	base := 0.03 * L // ~0.45s at ~15m (50ft) before nudges.

	materialFactor := 1.0
	switch roomWallMaterial {
	case "concrete", "glass":
		materialFactor = 1.25
	case "brick":
		materialFactor = 1.15
	case "wood", "panel", "paneling":
		materialFactor = 0.95
	case "curtain", "drape":
		materialFactor = 0.70
	case "acoustic", "foam", "tile":
		materialFactor = 0.45
	}

	// Reflectivity nudges: map ~0.70..0.96 reflectivity to ~0.8..1.2.
	reflect := roomWallReflect
	if reflect < 0 {
		reflect = 0
	} else if reflect > 1 {
		reflect = 1
	}
	reflectFactor := 0.8 + 0.4*((reflect-0.70)/0.26)
	if reflectFactor < 0.7 {
		reflectFactor = 0.7
	} else if reflectFactor > 1.3 {
		reflectFactor = 1.3
	}

	rt60 := base * materialFactor * reflectFactor

	// Bias shorter for the 2D ringing and lack of frequency-dependent losses.
	rt60 *= 0.75

	// Clamp to keep things sane for extreme world sizes.
	if rt60 < 0.20 {
		rt60 = 0.20
	} else if rt60 > 1.50 {
		rt60 = 1.50
	}
	return rt60
}
