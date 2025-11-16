package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

type levelFile struct {
	Version int      `json:"version"`
	Level   levelDef `json:"level"`
}

type levelDef struct {
	Name        string      `json:"name"`
	Size        levelSize   `json:"size"`
	PlayerStart levelPoint  `json:"player_start"`
	Exit        levelPoint  `json:"exit"`
	Walls       []levelRect `json:"walls"`
}

type levelSize struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

type levelPoint struct {
	X int `json:"x"`
	Y int `json:"y"`
}

type levelRect struct {
	ID string `json:"id"`
	X  int    `json:"x"`
	Y  int    `json:"y"`
	W  int    `json:"w"`
	H  int    `json:"h"`
}

func loadLevel(path string) (*levelDef, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var file levelFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, err
	}
	if file.Level.Size.Width <= 0 || file.Level.Size.Height <= 0 {
		return nil, fmt.Errorf("invalid level size %dx%d", file.Level.Size.Width, file.Level.Size.Height)
	}
	return &file.Level, nil
}

func (l *levelDef) startPosition(width, height int) (float64, float64) {
	if l == nil || width <= 0 || height <= 0 {
		return float64(width) / 2, float64(height) / 2
	}
	x := scaleToGrid(float64(l.PlayerStart.X), float64(l.Size.Width), float64(width))
	y := scaleToGrid(float64(l.PlayerStart.Y), float64(l.Size.Height), float64(height))
	return x, y
}

func (l *levelDef) fillWallMask(mask []bool, width, height int) {
	if l == nil || len(mask) != width*height || width <= 0 || height <= 0 {
		return
	}
	for i := range mask {
		mask[i] = false
	}
	for _, rect := range l.Walls {
		startX, spanX := scaleRange(rect.X, rect.W, l.Size.Width, width)
		startY, spanY := scaleRange(rect.Y, rect.H, l.Size.Height, height)
		for dy := 0; dy < spanY; dy++ {
			y := startY + dy
			if y < 0 || y >= height {
				continue
			}
			base := y * width
			for dx := 0; dx < spanX; dx++ {
				x := startX + dx
				if x < 0 || x >= width {
					continue
				}
				mask[base+x] = true
			}
		}
	}
}

func scaleToGrid(value, srcSize, dstSize float64) float64 {
	if srcSize <= 0 || dstSize <= 0 {
		return 0
	}
	return value / srcSize * dstSize
}

func scaleRange(start, length, srcSize, dstSize int) (int, int) {
	if srcSize <= 0 || dstSize <= 0 {
		return 0, 0
	}
	if length <= 0 {
		length = 1
	}
	scale := float64(dstSize) / float64(srcSize)
	startF := float64(start) * scale
	endF := float64(start+length) * scale
	startIdx := int(math.Floor(startF))
	endIdx := int(math.Ceil(endF))
	if startIdx < 0 {
		startIdx = 0
	}
	if endIdx > dstSize {
		endIdx = dstSize
	}
	if endIdx <= startIdx {
		if startIdx < dstSize {
			endIdx = startIdx + 1
		} else {
			startIdx = dstSize - 1
			endIdx = dstSize
		}
	}
	return startIdx, endIdx - startIdx
}
