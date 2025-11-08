package main

import (
	"os"
	"runtime/pprof"
	"sync"
)

// startDefaultPGORecording begins writing CPU profiles to the provided path.
func startDefaultPGORecording(path string) (func(), error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		f.Close()
		return nil, err
	}
	var once sync.Once
	stop := func() {
		once.Do(func() {
			pprof.StopCPUProfile()
			_ = f.Close()
		})
	}
	return stop, nil
}
