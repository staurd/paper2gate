`timescale 1ns / 1ps

module tb_butterfly;
    reg clk, rst_n;
    reg [11:0] a_in, b_in, w;
    wire [11:0] a_out, b_out;
    wire valid_out;

    butterfly_top dut (
        .clk(clk), .rst_n(rst_n),
        .a_in(a_in), .b_in(b_in), .w(w),
        .a_out(a_out), .b_out(b_out), .valid_out(valid_out)
    );

    always #5 clk = ~clk;  // 100 MHz

    initial begin
        clk = 0; rst_n = 0;
        a_in = 0; b_in = 0; w = 0;
        #15 rst_n = 1;

        // Test: a=1234, b=567, w=17
        // modmul computes a*b*k mod q, so twiddle must be pre-multiplied by k^{-1}
        // k^{-1} mod q = 3073 (since 13*3073 ≡ 1 mod 3329)
        // w' = 17 * 3073 mod 3329 = 2306
        // Expected: a_out = (1234 + 567*17) mod 3329 = 886
        // Expected: b_out = (1234 - 567*17) mod 3329 = 1582
        @(posedge clk);
        a_in = 1234; b_in = 567; w = 2306;

        // Wait for valid_out
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);  // extra wait

        if (valid_out) begin
            $display("a_out=%d (expected 886), b_out=%d (expected 1582)", a_out, b_out);
            if (a_out == 886 && b_out == 1582)
                $display("=== PASS ===");
            else
                $display("=== FAIL ===");
        end else begin
            $display("valid_out still low after 4 cycles");
        end

        $finish;
    end
endmodule
