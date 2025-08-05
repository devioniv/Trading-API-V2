import asyncio
import tkinter as tk
from tkinter import ttk
from ib_async import IB, util, Stock, Forex, Future, Option, Crypto, MarketOrder
import pandas as pd
import pandas_ta as ta

class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IBKR Trading Bot")
        self.root.geometry("800x600")

        self.ib = IB()
        self.ib.connectedEvent += self.on_connected
        self.ib.disconnectedEvent += self.on_disconnected
        self.ib.orderStatusEvent += self.on_order_status
        self.ib.execDetailsEvent += self.on_exec_details

        self.instruments = {}
        self.data = {}
        self.strategies = {}
        self.realtime_bars = {}

        self.create_widgets()

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill='both')

        # Connection buttons
        connection_frame = ttk.Frame(main_frame)
        connection_frame.pack(fill=tk.X, pady=5)
        self.connect_button = ttk.Button(connection_frame, text="Connect", command=self.connect_ib)
        self.connect_button.pack(side=tk.LEFT, padx=5)
        self.disconnect_button = ttk.Button(connection_frame, text="Disconnect", command=self.disconnect_ib, state=tk.DISABLED)
        self.disconnect_button.pack(side=tk.LEFT, padx=5)

        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(expand=True, fill='both', pady=5)

        # Add tabs
        self.instruments_tab = ttk.Frame(self.notebook)
        self.strategies_tab = ttk.Frame(self.notebook)
        self.logs_tab = ttk.Frame(self.notebook)
        self.summary_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.instruments_tab, text='Instruments')
        self.notebook.add(self.strategies_tab, text='Strategies')
        self.notebook.add(self.logs_tab, text='Logs')
        self.notebook.add(self.summary_tab, text='Summary')

        # Populate tabs
        self.create_instruments_tab()
        self.create_strategies_tab()
        self.create_logs_tab()
        self.create_summary_tab()

        # Status Bar
        self.status_bar = ttk.Label(self.root, text="Disconnected", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def connect_ib(self):
        self.connect_task = asyncio.create_task(self._connect_ib())

    async def _connect_ib(self):
        try:
            await self.ib.connectAsync('127.0.0.1', 7497, clientId=1)
        except Exception as e:
            self.log(f"Connection failed: {e}")

    def disconnect_ib(self):
        self.ib.disconnect()

    def on_connected(self):
        self.status_bar.config(text="Connected")
        self.connect_button.config(state=tk.DISABLED)
        self.disconnect_button.config(state=tk.NORMAL)
        self.log("Connected to IBKR")

    def on_disconnected(self):
        self.status_bar.config(text="Disconnected")
        self.connect_button.config(state=tk.NORMAL)
        self.disconnect_button.config(state=tk.DISABLED)
        self.log("Disconnected from IBKR")

    def on_order_status(self, trade):
        self.log(f"Order Status for {trade.contract.symbol}: {trade.orderStatus.status}")

    def on_exec_details(self, trade, fill):
        self.log(f"Execution for {trade.contract.symbol}: {fill.execution}")
        self.log_trade(trade, fill)

    def log_trade(self, trade, fill):
        summary = (
            f"--- TRADE SUMMARY ---\n"
            f"Symbol: {trade.contract.symbol}\n"
            f"Action: {trade.order.action}\n"
            f"Quantity: {trade.order.totalQuantity}\n"
            f"Avg Fill Price: {fill.execution.avgPrice}\n"
            f"Commission: {fill.commissionReport.commission}\n"
            f"---------------------\n"
        )
        self.summary_text.insert(tk.END, summary)
        self.summary_text.see(tk.END)

    def on_sec_type_selected(self, event):
        sec_type = self.sec_type_selector.get()
        if sec_type in ["OPT", "FUT"]:
            self.option_fut_frame.grid()
        else:
            self.option_fut_frame.grid_remove()

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        print(message)

    def create_instruments_tab(self):
        frame = ttk.LabelFrame(self.instruments_tab, text="Manage Instruments", padding="10")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Symbol:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.symbol_entry = ttk.Entry(frame)
        self.symbol_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="Sec Type:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.sec_type_selector = ttk.Combobox(frame, values=["STK", "CASH", "FUT", "OPT", "CRYPTO"])
        self.sec_type_selector.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.sec_type_selector.set("STK")
        self.sec_type_selector.bind("<<ComboboxSelected>>", self.on_sec_type_selected)

        # Frame for additional parameters for options and futures
        self.option_fut_frame = ttk.Frame(frame)
        self.option_fut_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(self.option_fut_frame, text="Expiry (YYYYMMDD):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.expiry_entry = ttk.Entry(self.option_fut_frame)
        self.expiry_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(self.option_fut_frame, text="Strike:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.strike_entry = ttk.Entry(self.option_fut_frame)
        self.strike_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(self.option_fut_frame, text="Right:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.right_selector = ttk.Combobox(self.option_fut_frame, values=["C", "P"])
        self.right_selector.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        self.right_selector.set("C")

        ttk.Label(self.option_fut_frame, text="Multiplier:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.multiplier_entry = ttk.Entry(self.option_fut_frame)
        self.multiplier_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)
        self.multiplier_entry.insert(0, "100")
        
        self.option_fut_frame.grid_remove() # Hide by default

        self.add_button = ttk.Button(frame, text="Add", command=self.add_instrument)
        self.add_button.grid(row=0, column=2, padx=5, pady=5)

        self.instrument_listbox = tk.Listbox(frame)
        self.instrument_listbox.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky=tk.NSEW)

        self.remove_button = ttk.Button(frame, text="Remove Selected", command=self.remove_instrument)
        self.remove_button.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(3, weight=1)

    def create_strategies_tab(self):
        frame = ttk.LabelFrame(self.strategies_tab, text="Configure Strategies", padding="10")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Instrument:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.strategy_instrument_selector = ttk.Combobox(frame)
        self.strategy_instrument_selector.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="Trade Quantity:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.quantity_entry = ttk.Entry(frame, width=10)
        self.quantity_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        self.quantity_entry.insert(0, "1")

        # SMA/WMA Strategy
        sma_frame = ttk.LabelFrame(frame, text="SMA/WMA Crossover", padding="10")
        sma_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        self.sma_wma_enabled = tk.BooleanVar()
        ttk.Checkbutton(sma_frame, text="Enable", variable=self.sma_wma_enabled).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(sma_frame, text="Short Period:").grid(row=1, column=0, sticky=tk.W)
        self.sma_short_period = ttk.Entry(sma_frame, width=5)
        self.sma_short_period.grid(row=1, column=1, sticky=tk.W)
        self.sma_short_period.insert(0, "10")
        ttk.Label(sma_frame, text="Long Period:").grid(row=2, column=0, sticky=tk.W)
        self.sma_long_period = ttk.Entry(sma_frame, width=5)
        self.sma_long_period.grid(row=2, column=1, sticky=tk.W)
        self.sma_long_period.insert(0, "50")

        # Trendline Breakout Strategy
        trendline_frame = ttk.LabelFrame(frame, text="Trendline Breakout", padding="10")
        trendline_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        self.trendline_enabled = tk.BooleanVar()
        ttk.Checkbutton(trendline_frame, text="Enable", variable=self.trendline_enabled).grid(row=0, column=0, sticky=tk.W)

        # Support/Resistance Strategy
        sr_frame = ttk.LabelFrame(frame, text="Support/Resistance", padding="10")
        sr_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        self.sr_enabled = tk.BooleanVar()
        ttk.Checkbutton(sr_frame, text="Enable", variable=self.sr_enabled).grid(row=0, column=0, sticky=tk.W)
        
        # Fair Value Gap Strategy
        fvg_frame = ttk.LabelFrame(frame, text="Fair Value Gap (FVG)", padding="10")
        fvg_frame.grid(row=1, column=2, rowspan=4, columnspan=2, padx=5, pady=5, sticky=tk.NSEW)
        self.fvg_enabled = tk.BooleanVar()
        ttk.Checkbutton(fvg_frame, text="Enable", variable=self.fvg_enabled).grid(row=0, column=0, sticky=tk.W, columnspan=2)

        ttk.Label(fvg_frame, text="Min Gap Size:").grid(row=1, column=0, sticky=tk.W)
        self.fvg_min_gap_size = ttk.Entry(fvg_frame, width=7)
        self.fvg_min_gap_size.grid(row=1, column=1, sticky=tk.W)
        self.fvg_min_gap_size.insert(0, "0.1")

        ttk.Label(fvg_frame, text="Max Candles:").grid(row=2, column=0, sticky=tk.W)
        self.fvg_max_candles = ttk.Entry(fvg_frame, width=7)
        self.fvg_max_candles.grid(row=2, column=1, sticky=tk.W)
        self.fvg_max_candles.insert(0, "10")

        ttk.Label(fvg_frame, text="S/R Lookback:").grid(row=3, column=0, sticky=tk.W)
        self.fvg_support_lookback = ttk.Entry(fvg_frame, width=7)
        self.fvg_support_lookback.grid(row=3, column=1, sticky=tk.W)
        self.fvg_support_lookback.insert(0, "10")
        
        self.fvg_use_sr = tk.BooleanVar(value=True)
        ttk.Checkbutton(fvg_frame, text="Use S/R", variable=self.fvg_use_sr).grid(row=4, column=0, sticky=tk.W)
        
        self.fvg_use_bos = tk.BooleanVar(value=True)
        ttk.Checkbutton(fvg_frame, text="Use BOS", variable=self.fvg_use_bos).grid(row=4, column=1, sticky=tk.W)

        self.fvg_use_regime = tk.BooleanVar(value=False)
        ttk.Checkbutton(fvg_frame, text="Use Regime Filter", variable=self.fvg_use_regime).grid(row=5, column=0, sticky=tk.W)

        self.fvg_use_dynamic_sizing = tk.BooleanVar(value=False)
        ttk.Checkbutton(fvg_frame, text="Dynamic Sizing", variable=self.fvg_use_dynamic_sizing).grid(row=5, column=1, sticky=tk.W)


        self.save_strategies_button = ttk.Button(frame, text="Save Strategies", command=self.save_strategies)
        self.save_strategies_button.grid(row=6, column=0, columnspan=4, pady=10)
        
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(3, weight=1)

    def create_logs_tab(self):
        frame = ttk.Frame(self.logs_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(frame, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

    def create_summary_tab(self):
        frame = ttk.Frame(self.summary_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        self.summary_text = tk.Text(frame, wrap=tk.WORD)
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_text.config(yscrollcommand=scrollbar.set)

    def add_instrument(self):
        symbol = self.symbol_entry.get().upper()
        if not symbol or symbol in self.instruments:
            return
        
        self.add_instrument_task = asyncio.create_task(self._add_instrument(symbol))

    async def _add_instrument(self, symbol):
        sec_type = self.sec_type_selector.get()
        if sec_type == "STK":
            contract = Stock(symbol, 'SMART', 'USD')
        elif sec_type == "CASH":
            contract = Forex(symbol)
        elif sec_type == "CRYPTO":
            if symbol.endswith('USD'):
                crypto_symbol = symbol[:-3]
                contract = Crypto(crypto_symbol, 'PAXOS', 'USD')
            else:
                self.log(f"Invalid crypto symbol: {symbol}. Please use the format 'BTCUSD', 'ETHUSD', etc.")
                return
        elif sec_type == "FUT":
            expiry = self.expiry_entry.get()
            multiplier = self.multiplier_entry.get()
            contract = Future(symbol, expiry, 'SMART', multiplier=multiplier, currency='USD')
        elif sec_type == "OPT":
            expiry = self.expiry_entry.get()
            strike = float(self.strike_entry.get())
            right = self.right_selector.get()
            multiplier = self.multiplier_entry.get()
            contract = Option(symbol, expiry, strike, right, 'SMART', multiplier=multiplier, currency='USD')
        else:
            self.log(f"Unsupported security type: {sec_type}")
            return

        try:
            qualified_contracts = await self.ib.qualifyContractsAsync(contract)
            if qualified_contracts:
                qualified_contract = qualified_contracts[0]
                self.instruments[symbol] = qualified_contract
                self.instrument_listbox.insert(tk.END, symbol)
                self.strategy_instrument_selector['values'] = list(self.instruments.keys())
                self.log(f"Added instrument: {symbol}")
                await self.fetch_historical_data(symbol)
            else:
                self.log(f"Could not qualify contract for {symbol}")
        except Exception as e:
            self.log(f"Error adding instrument {symbol}: {e}")

    def remove_instrument(self):
        selection = self.instrument_listbox.curselection()
        if not selection:
            return
        
        symbol = self.instrument_listbox.get(selection[0])
        
        bar_data_list = self.data.get(symbol)
        if bar_data_list:
            self.ib.cancelHistoricalData(bar_data_list)

        del self.instruments[symbol]
        if symbol in self.data:
            del self.data[symbol]
            
        self.instrument_listbox.delete(selection[0])
        self.strategy_instrument_selector['values'] = list(self.instruments.keys())
        self.log(f"Removed instrument: {symbol}")

    async def fetch_historical_data(self, symbol):
        contract = self.instruments.get(symbol)
        if not contract:
            return

        what_to_show = 'TRADES'
        if contract.secType == 'CRYPTO':
            what_to_show = 'AGGTRADES'
        elif contract.secType == 'CASH':
            what_to_show = 'MIDPOINT'

        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting='5 mins',
            whatToShow=what_to_show,
            useRTH=True,
            formatDate=1,
            keepUpToDate=True
        )
        if bars:
            self.data[symbol] = bars
            bars.updateEvent += self.on_bar_update
            self.log(f"Fetched {len(bars)} historical bars for {symbol}")
        else:
            self.log(f"Could not fetch historical data for {symbol}")

    def on_bar_update(self, bars, has_new_bar):
        if has_new_bar:
            symbol = bars.contract.symbol
            self.log(f"New 5-minute bar for {symbol}: {bars[-1]}")

            if symbol in self.strategies:
                for strategy in self.strategies[symbol]:
                    strategy.run()
    
    def save_strategies(self):
        symbol = self.strategy_instrument_selector.get()
        if not symbol:
            return

        try:
            quantity = int(self.quantity_entry.get())
        except ValueError:
            self.log("Invalid quantity. Please enter an integer.")
            return

        self.strategies[symbol] = []

        if self.sma_wma_enabled.get():
            try:
                short_period = int(self.sma_short_period.get())
                long_period = int(self.sma_long_period.get())
                strategy = SmaWmaCrossoverStrategy(self, symbol, short_period, long_period, quantity)
                self.strategies[symbol].append(strategy)
                self.log(f"SMA/WMA Crossover strategy enabled for {symbol}")
            except ValueError:
                self.log("Invalid SMA/WMA period. Please enter integers.")

        if self.trendline_enabled.get():
            strategy = TrendlineBreakoutStrategy(self, symbol, quantity=quantity)
            self.strategies[symbol].append(strategy)
            self.log(f"Trendline Breakout strategy enabled for {symbol}")

        if self.sr_enabled.get():
            strategy = SupportResistanceStrategy(self, symbol, quantity=quantity)
            self.strategies[symbol].append(strategy)
            self.log(f"Support/Resistance strategy enabled for {symbol}")

        if self.fvg_enabled.get():
            try:
                fvg_params = {
                    "quantity": quantity,
                    "min_gap_size": float(self.fvg_min_gap_size.get()),
                    "max_candles": int(self.fvg_max_candles.get()),
                    "support_lookback": int(self.fvg_support_lookback.get()),
                    "use_support_resistance": self.fvg_use_sr.get(),
                    "use_break_of_structure": self.fvg_use_bos.get(),
                    "use_market_regime_filter": self.fvg_use_regime.get(),
                    "use_dynamic_sizing": self.fvg_use_dynamic_sizing.get()
                }
                strategy = FairValueGapStrategy(self, symbol, **fvg_params)
                self.strategies[symbol].append(strategy)
                self.log(f"Fair Value Gap strategy enabled for {symbol}")
            except ValueError as e:
                self.log(f"Invalid FVG parameter: {e}")

def main():
    util.patchAsyncio()
    root = tk.Tk()
    TradingApp(root)

    async def run_tk():
        try:
            while True:
                root.update()
                root.update_idletasks()
                await asyncio.sleep(0.01)
        except tk.TclError:
            pass  # to avoid errors when the window is closed

    IB.run(run_tk())

class Strategy:
    def __init__(self, app, symbol):
        self.app = app
        self.symbol = symbol
        self.ib = app.ib
        self.contract = app.instruments.get(symbol)
        self.data = app.data.get(symbol)

    def run(self):
        raise NotImplementedError

class SmaWmaCrossoverStrategy(Strategy):
    def __init__(self, app, symbol, short_period, long_period, quantity):
        super().__init__(app, symbol)
        self.short_period = short_period
        self.long_period = long_period
        self.quantity = quantity

    def run(self):
        if self.data is None or len(self.data) < self.long_period:
            return

        df = util.df(self.data)
        df['short_sma'] = df['close'].rolling(window=self.short_period).mean()
        df['long_sma'] = df['close'].rolling(window=self.long_period).mean()

        # Check for crossover
        if len(df) > self.long_period:
            # Golden Cross
            if df['short_sma'].iloc[-2] < df['long_sma'].iloc[-2] and \
               df['short_sma'].iloc[-1] > df['long_sma'].iloc[-1]:
                order = MarketOrder('BUY', self.quantity)
                self.ib.placeOrder(self.contract, order)
                self.app.log(f"Golden Cross for {self.contract.symbol}. Placing BUY order.")

            # Death Cross
            elif df['short_sma'].iloc[-2] > df['long_sma'].iloc[-2] and \
                 df['short_sma'].iloc[-1] < df['long_sma'].iloc[-1]:
                order = MarketOrder('SELL', self.quantity)
                self.ib.placeOrder(self.contract, order)
                self.app.log(f"Death Cross for {self.contract.symbol}. Placing SELL order.")

class TrendlineBreakoutStrategy(Strategy):
    def __init__(self, app, symbol, period=20, quantity=1):
        super().__init__(app, symbol)
        self.period = period
        self.quantity = quantity

    def run(self):
        if self.data is None or len(self.data) < self.period:
            return

        df = util.df(self.data)
        if df is None or len(df) < self.period:
            return
            
        recent_data = df.tail(self.period)
        highs = recent_data['high']
        lows = recent_data['low']

        # Simplified trendline: a line connecting the highest high and lowest low
        resistance_slope = (highs.max() - highs.iloc[0]) / self.period
        resistance_intercept = highs.iloc[0]
        support_slope = (lows.min() - lows.iloc[0]) / self.period
        support_intercept = lows.iloc[0]

        current_price = df['close'].iloc[-1]
        
        # Simplified breakout logic
        resistance_price = resistance_slope * (self.period -1) + resistance_intercept
        support_price = support_slope * (self.period -1) + support_intercept

        if current_price > resistance_price:
            order = MarketOrder('BUY', self.quantity)
            self.ib.placeOrder(self.contract, order)
            self.app.log(f"Trendline breakout for {self.contract.symbol}. Placing BUY order.")

        elif current_price < support_price:
            order = MarketOrder('SELL', self.quantity)
            self.ib.placeOrder(self.contract, order)
            self.app.log(f"Trendline breakdown for {self.contract.symbol}. Placing SELL order.")

class SupportResistanceStrategy(Strategy):
    def __init__(self, app, symbol, period=50, quantity=1):
        super().__init__(app, symbol)
        self.period = period
        self.quantity = quantity

    def run(self):
        if self.data is None or len(self.data) < self.period:
            return

        df = util.df(self.data)
        if df is None or len(df) < self.period:
            return
            
        recent_data = df.tail(self.period)
        support_level = recent_data['low'].min()
        resistance_level = recent_data['high'].max()

        current_price = df['close'].iloc[-1]
        current_open = df['open'].iloc[-1]
        
        # Check for bounce from support
        if abs(current_price - support_level) < (support_level * 0.01): # Within 1% of support
            if current_price > current_open: # Bullish candle
                order = MarketOrder('BUY', self.quantity)
                self.ib.placeOrder(self.contract, order)
                self.app.log(f"Bounce from support for {self.contract.symbol}. Placing BUY order.")

        # Check for rejection from resistance
        elif abs(current_price - resistance_level) < (resistance_level * 0.01): # Within 1% of resistance
            if current_price < current_open: # Bearish candle
                order = MarketOrder('SELL', self.quantity)
                self.ib.placeOrder(self.contract, order)
                self.app.log(f"Rejection from resistance for {self.contract.symbol}. Placing SELL order.")

class FairValueGapStrategy(Strategy):
    def __init__(self, app, symbol, quantity=1, min_gap_size=0.1, max_candles=10, lookback_bos=20,
                 range_threshold=0.02, lower_threshold=0.02, upper_threshold=0.6,
                 support_lookback=10, support_tolerance=0.01, support_method="high_low_limit",
                 use_support_resistance=True, use_break_of_structure=True,
                 use_market_regime_filter=False, use_allowed_range_filter=True, use_dynamic_sizing=False):
        super().__init__(app, symbol)
        self.quantity = quantity
        self.min_gap_size = min_gap_size
        self.max_candles = max_candles
        self.lookback_bos = lookback_bos
        self.range_threshold = range_threshold
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.support_lookback = support_lookback
        self.support_tolerance = support_tolerance
        self.support_method = support_method
        self.use_support_resistance = use_support_resistance
        self.use_break_of_structure = use_break_of_structure
        self.use_market_regime_filter = use_market_regime_filter
        self.use_allowed_range_filter = use_allowed_range_filter
        self.use_dynamic_sizing = use_dynamic_sizing

        self.active_fvgs = []

    def run(self):
        if self.data is None or len(self.data) < max(self.lookback_bos, self.support_lookback, 200):
            return

        df = util.df(self.data)
        if df is None or len(df) < max(self.lookback_bos, self.support_lookback, 200):
            return
        
        # Calculate necessary indicators and confluences
        self.calculate_indicators(df)

        # Main loop logic (simplified for single pass on new bar)
        i = len(df) - 1
        if i < 2:
            return

        self.detect_new_fvg(df, i)
        self.update_active_fvgs(df, i)

    def calculate_indicators(self, df):
        if df is None:
            return
            
        if self.use_support_resistance:
            self.calculate_support_resistance(df)
        if self.use_market_regime_filter:
            self.detect_market_regime(df)
        if 'atr' not in df.columns: # Avoid recalculating
            atr_result = ta.atr(df['high'], df['low'], df['close'], length=14)
            if atr_result is not None:
                df['ATR'] = atr_result

    def detect_new_fvg(self, df, i):
        if df is None or len(df) <= i or i < 2:
            return
            
        high_i2, low_i = df['high'].iloc[i - 2], df['low'].iloc[i]
        low_i2, high_i = df['low'].iloc[i - 2], df['high'].iloc[i]

        fvg_types = [
            {"direction": "bullish", "fvg_low": high_i2, "fvg_high": low_i, "gap_size": low_i - high_i2, "threshold": high_i2 < low_i},
            {"direction": "bearish", "fvg_low": high_i, "fvg_high": low_i2, "gap_size": low_i2 - high_i, "threshold": low_i2 > high_i}
        ]

        for fvg in fvg_types:
            if fvg["gap_size"] >= self.min_gap_size and fvg["threshold"]:
                fvg_dict = {
                    "start_idx": i, "fvg_low": fvg["fvg_low"], "fvg_high": fvg["fvg_high"],
                    "zone_tested": False, "validation_candles": 0, "direction": fvg["direction"]
                }
                
                confluence_passed = True
                if self.use_allowed_range_filter and not self.FVG_is_in_allowed_range(df, fvg["fvg_low"], fvg["fvg_high"], i, direction=fvg["direction"]):
                    confluence_passed = False
                
                bos = self.is_break_of_structure(df, fvg["fvg_low"], fvg["fvg_high"], i, direction=fvg["direction"]) if self.use_break_of_structure else False
                
                sr_support = 'on_recent_support' in df.columns and df['on_recent_support'].iloc[i]
                sr_resistance = 'on_recent_resistance' in df.columns and df['on_recent_resistance'].iloc[i]
                
                sr = (sr_support if fvg['direction'] == 'bullish' else sr_resistance) if self.use_support_resistance else False

                if not (bos or sr) and (self.use_break_of_structure or self.use_support_resistance):
                    confluence_passed = False

                if confluence_passed:
                    self.active_fvgs.append(fvg_dict)

    def update_active_fvgs(self, df, i):
        if df is None or len(df) <= i:
            return
            
        current_candle = df.iloc[i]
        to_remove = []

        for idx, fvg_dict in enumerate(self.active_fvgs):
            fvg_low, fvg_high, direction = fvg_dict["fvg_low"], fvg_dict["fvg_high"], fvg_dict["direction"]

            if not fvg_dict['zone_tested']:
                if direction == "bullish" and (current_candle['low'] * (1 - self.support_tolerance) <= fvg_high <= current_candle['high'] * (1 + self.support_tolerance)):
                    fvg_dict['zone_tested'] = True
                elif direction == "bearish" and (current_candle['low'] <= fvg_low <= current_candle['high']):
                    fvg_dict['zone_tested'] = True
            else:
                fvg_dict['validation_candles'] += 1
                if fvg_dict['validation_candles'] >= self.max_candles:
                    to_remove.append(idx)
                    continue

                signal = 0
                if direction == "bullish" and current_candle['close'] > fvg_high:
                    signal = 1
                elif direction == "bearish" and current_candle['close'] < fvg_low:
                    signal = -1

                if self.use_market_regime_filter and 'is_trending' in df.columns and not df['is_trending'].iloc[i]:
                    signal = 0

                if signal != 0:
                    quantity = self.determine_position_size(df, i) if self.use_dynamic_sizing else self.quantity
                    action = 'BUY' if signal == 1 else 'SELL'
                    order = MarketOrder(action, quantity)
                    self.ib.placeOrder(self.contract, order)
                    self.app.log(f"FVG Signal for {self.contract.symbol}. Placing {action} order of {quantity} shares.")
                    to_remove.append(idx)

        for r_idx in sorted(to_remove, reverse=True):
            if r_idx < len(self.active_fvgs):
                self.active_fvgs.pop(r_idx)
            
    def calculate_support_resistance(self, df):
        if df is None or len(df) < self.support_lookback:
            return
            
        if self.support_method == "slope":
            df['ma'] = df['close'].rolling(window=5).mean()
            df['slope'] = df['ma'].diff()
            df['is_support_candle'] = (df['slope'].shift(1) < 0) & (df['slope'] > 0)
            df['is_resistance_candle'] = (df['slope'].shift(1) > 0) & (df['slope'] < 0)
        else:
            n = self.support_lookback
            df['is_support_candle'] = df['low'] == df['low'].rolling(2*n+1, center=True).min()
            df['is_resistance_candle'] = df['high'] == df['high'].rolling(2*n+1, center=True).max()

        support_prices = df['low'][df['is_support_candle']]
        resistance_prices = df['high'][df['is_resistance_candle']]

        # Initialize columns if they don't exist
        if 'on_recent_support' not in df.columns:
            df['on_recent_support'] = False
        if 'on_recent_resistance' not in df.columns:
            df['on_recent_resistance'] = False

        # Calculate support/resistance using a more efficient approach
        for idx in range(len(df)):
            if idx < self.support_lookback:
                continue
                
            current_low = df['low'].iloc[idx]
            current_high = df['high'].iloc[idx]
            
            # Check for support
            for support_price in support_prices:
                if support_price < current_low and abs(current_low - support_price) < self.support_tolerance * support_price:
                    df.loc[df.index[idx], 'on_recent_support'] = True
                    break
            
            # Check for resistance  
            for resistance_price in resistance_prices:
                if resistance_price > current_high and abs(current_high - resistance_price) < self.support_tolerance * resistance_price:
                    df.loc[df.index[idx], 'on_recent_resistance'] = True
                    break

    def is_break_of_structure(self, df, fvg_low, fvg_high, current_idx, direction="bullish"):
        if df is None or len(df) <= current_idx:
            return False
            
        start_idx = max(0, current_idx - self.lookback_bos)
        recent_df = df.iloc[start_idx:current_idx]
        current_price = df['close'].iloc[current_idx]

        if direction == "bullish":
            swing_high = recent_df['high'].max()
            if current_price >= swing_high and (fvg_low >= swing_high or abs(fvg_low - swing_high) <= self.range_threshold * swing_high):
                return True
        else:
            swing_low = recent_df['low'].min()
            if current_price <= swing_low and (fvg_high <= swing_low or abs(fvg_high - swing_low) <= self.range_threshold * swing_low):
                return True
        return False

    def FVG_is_in_allowed_range(self, df, fvg_low, fvg_high, current_idx, lookback=200, direction="bullish"):
        if df is None or len(df) <= current_idx:
            return False
            
        start_idx = max(0, current_idx - lookback)
        recent_df = df.iloc[start_idx:current_idx + 1]
        range_high, range_low = recent_df['high'].max(), recent_df['low'].min()
        range_height = range_high - range_low
        if range_height <= 0: 
            return False

        fvg_midpoint = (fvg_low + fvg_high) / 2
        fvg_position = (fvg_midpoint - range_low) / range_height
        
        return self.lower_threshold <= fvg_position <= self.upper_threshold
        
    def detect_market_regime(self, df, ma_length=50, slope_threshold=0.01, atr_length=14, atr_threshold=1.5, adx_length=14, adx_threshold=25):
        if df is None or len(df) < ma_length:
            return
            
        df['ma'] = df['close'].rolling(window=ma_length).mean()
        df['ma_slope'] = df['ma'].diff()
        df['is_strong_bullish_slope'] = df['ma_slope'] > slope_threshold
        df['is_strong_bearish_slope'] = df['ma_slope'] < -slope_threshold
        
        atr = ta.atr(df['high'], df['low'], df['close'], length=atr_length)
        if atr is not None and not atr.empty:
            df['ATR'] = atr
            df['atr_mean'] = df['ATR'].rolling(atr_length).mean()
            df['high_volatility'] = df['ATR'] > df['atr_mean'] * atr_threshold
        else:
            df['high_volatility'] = False

        adx_df = ta.adx(df['high'], df['low'], df['close'], length=adx_length)
        if adx_df is not None and not adx_df.empty and f'ADX_{adx_length}' in adx_df.columns:
            df['adx'] = adx_df[f'ADX_{adx_length}']
            df['strong_adx'] = df['adx'] > adx_threshold
        else:
            df['strong_adx'] = False

        df['is_trending'] = (df['is_strong_bullish_slope'] | df['is_strong_bearish_slope']) & df['high_volatility'] & df['strong_adx']

    def determine_position_size(self, df, i):
        if df is None or len(df) <= i:
            return self.quantity
            
        confluence_score = 0
        if self.use_support_resistance and 'on_recent_support' in df.columns and (df['on_recent_support'].iloc[i] or df['on_recent_resistance'].iloc[i]):
            confluence_score += 1
        if self.use_break_of_structure:
            is_bos = self.is_break_of_structure(df, df['low'].iloc[i-2], df['high'].iloc[i], i)
            if is_bos:
                confluence_score += 1
        if self.use_market_regime_filter and 'is_trending' in df.columns and df['is_trending'].iloc[i]:
            confluence_score += 1

        position_sizes = {1: 0.5, 2: 1, 3: 1.5}
        size_multiplier = position_sizes.get(confluence_score, 0.5)
        return max(1, int(self.quantity * size_multiplier))

if __name__ == '__main__':
    main()
