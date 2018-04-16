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


class Darius(object):
    def __init__(self, ifm_height, ifm_width, ifm_depth, kernel_height,
                 kernel_width, pad, stride, channels,
                 pool_kernel_height, pool_kernel_width, pool_stride,
                 ifm_baseaddr, weights_baseaddr, ofm_baseaddr):
        """Return a new Convolution with Maxpool object"""

        self.ifm_height = ifm_height
        self.ifm_width = ifm_width
        self.ifm_depth = ifm_depth
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.pad = pad
        self.stride = stride
        self.channels = channels
        self.pool_kernel_height = pool_kernel_height
        self.pool_kernel_width = pool_kernel_width
        self.pool_stride = pool_stride
        self.ifm_baseaddr = ifm_baseaddr
        self.weights_baseaddr = weights_baseaddr
        self.ofm_baseaddr = ofm_baseaddr

        def derive_attributes():
            self.ofm_height = ceil((self.ifm_height + 2 * self.pad - self.kernel_height) / self.stride + 1)
            self.ofm_width = ceil((self.ifm_width + 2 * self.pad - self.kernel_width) / self.stride + 1)
            self.ofm_depth = self.channels
            self.ifm_slices = ceil(self.ifm_depth / C_NUM_OF_ROWS)
            self.ofm_slices = ceil(self.channels / C_NUM_OF_COLS)
            self.ofm_fragments = 1
            self.ifm_mem_fragments = 1
        
            self.ifm_packet_length = self.ifm_width * self.ifm_height * self.ifm_slices
            self.ifm_depth_offset = self.ifm_width * self.ifm_height * self.ifm_depth
            self.ifm_height_offset = 0
            
            self.ofm_offset = self.ofm_height * self.ofm_width * self.ofm_depth

            self.weights_packet_length = self.kernel_height * self.kernel_width * self.ifm_depth
            self.weight_depth_offset = self.kernel_height * self.kernel_width * self.ifm_depth * \
                              C_NUM_OF_COLS
            self.weight_offset = self.kernel_height * self.kernel_width * self.ifm_depth
            self.weight_pkt_offset = C_NUM_OF_ROWS * self.kernel_height * self.kernel_width
        
            self.reserved = 0
            
            self.pool_input_height = self.ofm_height
            self.pool_input_width = self.ofm_width
            # divid by one so to make it float calculation
            try:
                self.pool_output_height = ceil(((self.pool_input_height - self.pool_kernel_height) / 1.0) / self.pool_stride / 1.0 + 1)
                self.pool_output_width = ceil(((self.pool_input_width - self.pool_kernel_width) / 1.0) / self.pool_stride / 1.0 + 1)
            except ZeroDivisionError:
                self.pool_output_height = 0
                self.pool_output_width = 0
                print("INFO: POOL STRIDE OF 0 DISABLES MAXPOOLING; ONLY CONVOLUTION WOULD HAPPEN!")

            # pool_stride has to be a multiple of 2
            if (self.pool_stride != 0 \
                and self.pool_output_height > 5 and self.pool_output_width > 5 \
                and self.pool_output_width * self.pool_kernel_width < 1 << 9 \
                and self.pool_output_width < 1 << 8):

                # WHEN MAXPOOL IS ENABLED, THE OUTPUT SIZE WILL BE SMALLER
                # THERFORE, THE OFM_PACKET_LENGTH HAS TO BE ADJUSTED ACCORDINGLY
                self.ofm_packet_length = self.pool_output_height * self.pool_output_width * self.ofm_slices
            else:
                self.pool_output_height = 0
                self.pool_output_width = 0
                self.pool_kernel_height = 0
                self.pool_kernel_width = 0
                self.pool_stride = 0
                self.ofm_packet_length = self.ofm_height * self.ofm_width * self.ofm_slices   
   
        derive_attributes()

    def IP_cmd(self):        
        """ Construct convolution command for CNNDataflow IP if the arguments
        inputed are in supported range of Convolution Overlay """

        # The IFM demensions to be in range (6,32)
        if (self.ifm_height < 6 or self.ifm_width < 6 or self.ifm_height > 32 or self.ifm_width > 32):
            print("ERROR: THE IFM VOLUME IS EITHER SMALLER/LARGER THAN SUPPORTED")
            print("TIP: Make sure IFM height and width are in range from 6 to 32")
            return False

        # The IFM depth to be multiples of 8 and are in range (8,1024)
        if (self.ifm_depth <= 512 or self.ifm_depth >= 8):
            if (self.ifm_depth % 8 != 0):
                print("ERROR: THE IFM DEPTH NEEDS TO BE IN MULTIPLES OF 8 IN THE RANGE 8 TO 512")
                return False
        else:
            print("ERROR: THE IFM DEPTH NEEDS TO BE IN MULTIPLES OF 8 IN THE RANGE 8 TO 512")
            return False

        # The Kernel demensions to be in range (1,16)
        if (self.kernel_height < 1 or self.kernel_width < 1 or self.kernel_height > 16 or self.kernel_width > 16):
            print("ERROR: THE KERNEL DIMENSIONS ARE EITHER SMALLER/LARGER THAN SUPPORTED")
            print("TIP: Make sure Kernel height and width are in range from 1 to 16")
            return False

        if (self.stride > 4 or self.stride == 0 or (self.stride != 1 and self.stride % 2 != 0)):
            print("ERROR: THIS STRIDE FOR CONVOLUTION IS NOT RECOMMENDED")
            print("TIP: Make sure stride is either 1, 2 and 4")
            return False

        # The Number of Pad bits to be in range (0,16)
        if (self.pad < 0 or self.pad > 16):
            print("ERROR: THE PADDED BITS ARE EITHER SMALLER/LARGER THAN SUPPORTED")
            print("TIP: Make sure Pad is in range from 0 to 16")
            return False

        # The OFM Channels to be multiples of 8 and are in range (8,1024)
        if (self.ofm_depth <= 512 or self.ofm_depth >= 8):
            if (self.ofm_depth % 8 != 0):
                print("ERROR: THE NUMBER OF CHANNELS NEEDS TO BE IN MULTIPLES OF 8 IN THE RANGE 8 TO 512")
                return False
        else:
            print("ERROR: THE NUMBER OF CHANNELS NEEDS TO BE IN MULTIPLES OF 8 IN THE RANGE 8 TO 512")
            return False

        # The accumulation loopback has 10 cycle delay
        if (self.ofm_height * self.ofm_width < 10):
            print("ERROR: THE OFM VOLUME IS SMALLER THAN SUPPORTED")
            print("TIP: Manage the IFM dimensions, kernel dimensions and other "
                  "arguments such that ofm volume is of moderate size ")
            return False

        # The 2D dimensions are limited by BRAM chosen
        if (self.ifm_height * self.ifm_width > (1 << C_MAX_ADDR_WIDTH) or self.ofm_height * self.ofm_width > (1 << C_MAX_ADDR_WIDTH)):
            print("ERROR: THE IFM/OFM PLANE DOES NOT FIT IN THE LINE BUFFER")
            return False

        # The max allowable block read (BTT) by the datamover is limited by
        # 2^23. The num of channels is currently limited by this number
        if (self.ofm_height * self.ofm_width * self.channels * (C_MAX_INPUT_WIDTH / 8) > 1 << 23):
            print("ERROR: THE NUMBER OF CHANNELS IS LARGER THAN THE MAXIMUM "
                  "ALLOWABLE BYTES-TO-TRANSFER(BTT) OF DATAMOVER")
            print("TIP: Decrease the number of channels")
            return False

        while True:
            print("All IP arguments are in supported range")
            cmd_conv = np.array([self.ifm_height, self.ifm_width, self.kernel_height,
                                 self.kernel_width, self.stride, self.pad, self.ofm_height,
                                 self.ofm_width, self.ifm_slices, self.ofm_slices,
                                 self.ofm_fragments, self.ifm_mem_fragments],
                                dtype='uint16')

            cmd_addr = np.array([self.ifm_baseaddr, self.ifm_packet_length,
                                 self.ifm_depth_offset, self.ifm_height_offset,
                                 self.ofm_baseaddr, self.ofm_packet_length,
                                 self.weights_baseaddr, self.weights_packet_length,
                                 self.weight_depth_offset],
                                dtype='uint32')

            cmd_mode = np.array([0, 0], dtype='uint16')

            cmd_pool = np.array([self.pool_input_height, self.pool_input_width,
                                 self.pool_kernel_height, self.pool_kernel_width,
                                 self.pool_output_height, self.pool_output_width,
                                 self.pool_stride, 0],
                                dtype='uint16')

            cmd_rsvd = np.zeros((12,), dtype='uint32')

            IP_cmd   = cmd_conv.tobytes() + \
                       cmd_addr.tobytes() + \
                       cmd_mode.tobytes() + \
                       cmd_pool.tobytes() + \
                       cmd_rsvd.tobytes()
            return IP_cmd
            break

    def reshape_and_copy_ifm(self, ifm_sw, ifm):
        """ Reshape the IFM Volume as per IP requirement and copy to physical
        memory with ifm pointer """
        hw_index = 0
        for i in range(0, self.ifm_slices):
            for j in range(0, self.ifm_height * self.ifm_width):
                for k in range(0, C_NUM_OF_ROWS):
                    index = i * (self.ifm_height * self.ifm_width) * C_NUM_OF_ROWS + \
                            k * (self.ifm_height * self.ifm_width) + j
                    ifm[hw_index] = ifm_sw[index]
                    hw_index = hw_index + 1

    def reshape_and_copy_weights(self, weights_sw, weights):
        """ Reshape the Weights as per IP requirement and copy to physical
        memory with weights pointer """
        weights_index = 0
        for i in range(0, self.ofm_slices):
            for j in range(0, self.ifm_slices):
                for k in range(0, self.kernel_height * self.kernel_width):
                    for r in range(0, C_NUM_OF_ROWS):
                        for c in range(0, C_NUM_OF_COLS):
                            addr = i * C_NUM_OF_COLS * self.weight_offset + \
                                   c * self.weight_offset + \
                                   j * self.weight_pkt_offset + \
                                   r * self.kernel_height * self.kernel_width + \
                                   k
                            weights[weights_index] = weights_sw[addr]
                            weights_index = weights_index + 1

    def calc_efficiency(self, hw_cycles):
        """ Gives efficiency of IP """
        num_of_calc = self.ofm_height * self.ofm_width * \
                      self.ofm_depth * self.kernel_height * self.kernel_width * self.ifm_depth
        theoretical_cycles = num_of_calc / (C_NUM_OF_COLS * C_NUM_OF_ROWS)
        efficiency = (theoretical_cycles / hw_cycles) * 100
        return float(efficiency)
            