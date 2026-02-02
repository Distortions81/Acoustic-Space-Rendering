//go:build cl10
// +build cl10

package cl

func (k *Kernel) ArgName(index int) (string, error) {
	return "", ErrUnsupported
}
