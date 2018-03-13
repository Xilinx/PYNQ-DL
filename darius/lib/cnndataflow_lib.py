#   Copyright (c) 2018, Xilinx, Inc.
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   1.  Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#   2.  Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#   3.  Neither the name of the copyright holder nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pynq import MMIO
from pynq import Xlnk
import numpy as np
from math import *
from random import *

__author__ = "Ehsan Ghasemi; Radhika Pokanati"
__copyright__ = "Copyright 2016, Xilinx"
__email__ = "pynq_support@xilinx.com"

# CNNDataflow IP Constants
C_MAX_ADDR_WIDTH = 12
C_MAX_ITER_WIDTH = 6
C_MAX_IMG_DIMENSION_WIDTH = 10
C_MAX_INPUT_WIDTH = 16
C_NUM_OF_ROWS = 8
C_NUM_OF_COLS = 8

mmu = Xlnk()
SIZE = 5000000  # 20 MB of numpy.uint32s


class CNNDataflow(object):
    def __init__(self, ifm_height, ifm_width, ifm_depth, kernel_height,
                 kernel_width, pad, stride, channels, ifm_baseaddr,
                 weights_baseaddr, ofm_baseaddr):
        """Return a new Convolution object"""

        self.ifm_height = ifm_height
        self.ifm_width = ifm_width
        self.ifm_depth = ifm_depth
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.pad = pad
        self.stride = stride
        self.channels = channels
        self.ifm_baseaddr = ifm_baseaddr
        self.weights_baseaddr = weights_baseaddr
        self.ofm_baseaddr = ofm_baseaddr

        global ofm_height
        global ofm_width
        global ofm_depth
        global ifm_slices
        global ofm_slices
        global ofm_fragments
        global ifm_mem_fragments
        global ifm_packet_length
        global ifm_depth_offset
        global ifm_height_offset
        global ofm_packet_length
        global ofm_offset
        global weights_packet_length
        global weight_depth_offset
        global weight_offset
        global weight_pkt_offset
        global reserved

        ofm_height = ceil((ifm_height + 2 * pad - kernel_height) / stride + 1)
        ofm_width = ceil((ifm_width + 2 * pad - kernel_width) / stride + 1)
        ofm_depth = channels
        ifm_slices = ceil(ifm_depth / C_NUM_OF_ROWS)
        ofm_slices = ceil(ofm_depth / C_NUM_OF_COLS)
        ofm_fragments = 1
        ifm_mem_fragments = 1
        ifm_packet_length = ifm_width * ifm_height * ifm_slices
        ifm_depth_offset = ifm_width * ifm_height * ifm_depth
        ifm_height_offset = 0
        ofm_packet_length = ofm_height * ofm_width * ofm_slices
        ofm_offset = ofm_height * ofm_width * ofm_depth
        weights_packet_length = kernel_height * kernel_width * ifm_depth
        weight_depth_offset = kernel_height * kernel_width * ifm_depth * \
                              C_NUM_OF_COLS
        weight_offset = kernel_height * kernel_width * ifm_depth
        weight_pkt_offset = C_NUM_OF_ROWS * kernel_height * kernel_width
        reserved = 0

    def construct_conv_cmd(self, ifm_height, ifm_width, ifm_depth,
                           kernel_height, kernel_width, pad, stride, channels,
                           ifm_baseaddr, weights_baseaddr, ofm_baseaddr):
        """ Construct convolution command for CNNDataflow IP if the arguments 
        inputed are in supported range of Convolution Overlay """

        # The IFM demensions to be in range (6,32)
        if (ifm_height < 6 or ifm_width < 6 or
                    ifm_height > 32 or ifm_width > 32):
            print(
                "ERROR: THE IFM VOLUME IS EITHER SMALLER/LARGER THAN SUPPORTED")
            print(
                "TIP: Make sure IFM height and width are in range from 6 to 32")
            return False

        # The IFM depth to be multiples of 8 and are in range (8,1024)
        if (ifm_depth <= 1024 or ifm_depth >= 8):
            if (ifm_depth % 8 != 0):
                print(
                    "ERROR: THE IFM DEPTH NEEDS TO BE IN MULTIPLES OF 8 IN "
                    "THE RANGE 8 TO 1024")
                return False
        else:
            print(
                "ERROR: THE IFM DEPTH NEEDS TO BE IN MULTIPLES OF 8 IN THE "
                "RANGE 8 TO 1024")
            return False

        # The Kernel demensions to be in range (1,16)
        if (kernel_height < 1 or kernel_width < 1 or
                    kernel_height > 16 or kernel_width > 16):
            print(
                "ERROR: THE KERNEL DIMENSIONS ARE EITHER SMALLER/LARGER "
                "THAN SUPPORTED")
            print(
                "TIP: Make sure Kernel height and width are in range from "
                "1 to 16")
            return False

        # The common strides are 0,1,2 and 4)
        if (stride > 4 or (stride != 1 and stride % 2 != 0)):
            print("ERROR: THIS STRIDE IS NOT RECOMMENDED")
            print("TIP: Make sure stride is either 0,1,2 and 4")
            return False

        # The Number of Pad bits to be in range (0,16)
        if (pad < 0 or pad > 16):
            print(
                "ERROR: THE PADDED BITS ARE EITHER SMALLER/LARGER THAN "
                "SUPPORTED")
            print("TIP: Make sure Pad is in range from 1 to 16")
            return False

        # The OFM Channels to be multiples of 8 and are in range (8,1024)
        if (ofm_depth <= 1024 or ofm_depth >= 8):
            if (ofm_depth % 8 != 0):
                print(
                    "ERROR: THE NUMBER OF CHANNELS NEEDS TO BE IN MULTIPLES "
                    "OF 8 IN THE RANGE 8 TO 1024")
                return False
        else:
            print(
                "ERROR: THE NUMBER OF CHANNELS NEEDS TO BE IN MULTIPLES OF "
                "8 IN THE RANGE 8 TO 1024")
            return False

        # The accumulation loopback has 10 cycle delay
        if (ofm_height * ofm_width < 10):
            print("ERROR: THE OFM VOLUME IS SMALLER THAN SUPPORTED")
            print(
                "TIP: If the IFM dimensions are in supported range, check "
                "the kernel dimensions and other arguments")
            return False

        # The 2D dimensions are limited by BRAM chosen
        if (ifm_height * ifm_width > (
                    1 << C_MAX_ADDR_WIDTH) or ofm_height * ofm_width > (
                    1 << C_MAX_ADDR_WIDTH)):
            print("ERROR: THE IFM/OFM PLANE DOES NOT FIT IN THE LINE BUFFER")
            return False

        if (ceil(log2(ifm_slices)) > C_MAX_ITER_WIDTH or ceil(
                log2(ofm_slices)) > C_MAX_ITER_WIDTH):
            print(
                "ERROR: THE MAXIMUM ITERATION BITWIDTH IS SMALLER THAN GIVEN "
                "IFM_SLICES/OFM_SLICES")
            print(
                "TIP: INCREASE THE \"C_MAX_ITER_WIDTH\" PARAMETER IF NECESSARY")
            return False

        # The max allowable block read (BTT) by the datamover is limited by 
        # 2^23. The num of channels is currently limited by this number
        if (ofm_height * ofm_width * channels * (
                    C_MAX_INPUT_WIDTH / 8) > 1 << 23):
            print(
                "ERROR: THE NUMBER OF CHANNELS IS LARGER THAN THE MAXIMUM "
                "ALLOWABLE BYTES-TO-TRANSFER(BTT) OF DATAMOVER")
            print("TIP: Decrease the number of channels")
            return False

        while True:
            print("All convolution arguments are in supported range")
            conv_cmd_shorts = np.array([ifm_height, ifm_width, kernel_height,
                                        kernel_width, stride, pad, ofm_height,
                                        ofm_width, ifm_slices, ofm_slices,
                                        ofm_fragments, ifm_mem_fragments],
                                       dtype='uint16')

            conv_cmd_ints = np.array([ifm_baseaddr, ifm_packet_length,
                                      ifm_depth_offset, ifm_height_offset,
                                      ofm_baseaddr, ofm_packet_length,
                                      weights_baseaddr, weights_packet_length,
                                      weight_depth_offset, reserved],
                                     dtype='uint32')

            global conv_cmd

            conv_cmd = conv_cmd_shorts.tobytes() + conv_cmd_ints.tobytes()
            conv_cmd

            print("Convolution command to CNNDataflow IP: ")
            print(str(conv_cmd))
            break

    def reshape_and_copy_ifm(self, ifm_height, ifm_width, ifm_depth, ifm_sw,
                             ifm):
        """ Reshape the IFM Volume as per IP requirement and copy to physical 
        memory with ifm pointer """
        hw_index = 0
        for i in range(0, ifm_slices):
            for j in range(0, ifm_height * ifm_width):
                for k in range(0, C_NUM_OF_ROWS):
                    index = i * (
                        ifm_height * ifm_width) * C_NUM_OF_ROWS + k * (
                        ifm_height * ifm_width) + j
                    ifm[hw_index] = ifm_sw[index]
                    hw_index = hw_index + 1
                    # return ifm

    def reshape_and_copy_weights(self, kernel_height, kernel_width, ifm_depth,
                                 weights_sw, weights):
        """ Reshape the Weights as per IP requirement and copy to physical 
        memory with weights pointer """
        weights_index = 0
        for i in range(0, ofm_slices):
            for j in range(0, ifm_slices):
                for k in range(0, kernel_height * kernel_width):
                    for r in range(0, C_NUM_OF_ROWS):
                        for c in range(0, C_NUM_OF_COLS):
                            addr = i * C_NUM_OF_COLS * \
                                   weight_offset + \
                                   c * weight_offset + \
                                   j * weight_pkt_offset + \
                                   r * kernel_height * kernel_width + k
                            weights[weights_index] = weights_sw[addr]
                            weights_index = weights_index + 1
                            # return weights

    def load_conv_cmd(self, cmd_baseaddr):
        """ Load the convolution command to the command physical address """
        cmd_mem = MMIO(cmd_baseaddr, SIZE)
        cmd_mem.write(0x0, conv_cmd)

    # for i in range(0, 64, 4):
    # print(hex(cmd_mem.read(i)))

    def calc_efficiency(self, kernel_height, kernel_width, ifm_depth,
                        hw_cycles):
        num_of_calc = ofm_height * ofm_width * \
                      ofm_depth * kernel_height * kernel_width * ifm_depth
        theoretical_cycles = num_of_calc / (C_NUM_OF_COLS * C_NUM_OF_ROWS)
        efficiency = (theoretical_cycles / hw_cycles) * 100
        return float(efficiency)
