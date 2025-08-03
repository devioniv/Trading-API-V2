import asyncio
import tkinter as tk
from tkinter import ttk
from ib_async import IB, util, Stock, Forex, Future, Option, Crypto, MarketOrder
import pandas as pd

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
        asyncio.create_task(self._connect_ib())

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

        self.save_strategies_button = ttk.Button(frame, text="Save Strategies", command=self.save_strategies)
        self.save_strategies_button.grid(row=4, column=0, columnspan=2, pady=10)
        
        frame.grid_columnconfigure(1, weight=1)

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
        
        asyncio.create_task(self._add_instrument(symbol))

    async def _add_instrument(self, symbol):
        sec_type = self.sec_type_selector.get()
        if sec_type == "STK":
            contract = Stock(symbol, 'SMART', 'USD')
        elif sec_type == "CASH":
            contract = Forex(symbol)
        elif sec_type == "CRYPTO":
            if len(symbol) > 3:
                crypto_symbol = symbol[:-3]
                currency = symbol[-3:]
                contract = Crypto(crypto_symbol, 'PAXOS', currency)
            else:
                self.log(f"Invalid crypto symbol: {symbol}. Please use the format like 'BTCUSD'.")
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
                self.start_realtime_updates(symbol)
            else:
                self.log(f"Could not qualify contract for {symbol}")
        except Exception as e:
            self.log(f"Error adding instrument {symbol}: {e}")

    def remove_instrument(self):
        selection = self.instrument_listbox.curselection()
        if not selection:
            return
        
        symbol = self.instrument_listbox.get(selection[0])
        
        rt_bars = self.realtime_bars.get(symbol)
        if rt_bars:
            self.ib.cancelRealTimeBars(rt_bars)
            del self.realtime_bars[symbol]
        
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

        # Fetch 1 day of 1-minute bars
        what_to_show = 'TRADES'
        if contract.secType == 'CRYPTO':
            what_to_show = 'AGGTRADES'
        elif contract.secType == 'CASH':
            what_to_show = 'MIDPOINT'

        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting='1 min',
            whatToShow=what_to_show,
            useRTH=True,
            formatDate=1
        )
        if bars:
            df = util.df(bars)
            self.data[symbol] = df
            self.log(f"Fetched {len(bars)} historical bars for {symbol}")
            self.start_realtime_updates(symbol)
        else:
            self.log(f"Could not fetch historical data for {symbol}")

    def start_realtime_updates(self, symbol):
        contract = self.instruments.get(symbol)
        if not contract:
            return
        
        rt_bars = self.ib.reqRealTimeBars(contract, 5, 'TRADES', False, [])
        self.realtime_bars[symbol] = rt_bars
        rt_bars.updateEvent += self.on_bar_update

    def on_bar_update(self, bars, has_new_bar):
        if has_new_bar:
            symbol = bars.contract.symbol
            df = self.data.get(symbol)
            if df is not None:
                bar_data = {
                    'date': pd.to_datetime(bars[-1].time),
                    'open': bars[-1].open_,
                    'high': bars[-1].high,
                    'low': bars[-1].low,
                    'close': bars[-1].close,
                    'volume': bars[-1].volume,
                    'average': bars[-1].wap,
                    'barCount': bars[-1].count
                }
                new_row = pd.DataFrame([bar_data]).set_index('date')
                self.data[symbol] = pd.concat([df, new_row])
                self.log(f"New bar for {symbol}: {bars[-1]}")

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

def main():
    util.patchAsyncio()
    root = tk.Tk()
    app = TradingApp(root)

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

        # Calculate SMAs
        self.data['short_sma'] = self.data['close'].rolling(window=self.short_period).mean()
        self.data['long_sma'] = self.data['close'].rolling(window=self.long_period).mean()

        # Check for crossover
        if len(self.data) > self.long_period:
            # Golden Cross
            if self.data['short_sma'].iloc[-2] < self.data['long_sma'].iloc[-2] and \
               self.data['short_sma'].iloc[-1] > self.data['long_sma'].iloc[-1]:
                order = MarketOrder('BUY', self.quantity)
                self.ib.placeOrder(self.contract, order)
                self.app.log(f"Golden Cross for {self.contract.symbol}. Placing BUY order.")

            # Death Cross
            elif self.data['short_sma'].iloc[-2] > self.data['long_sma'].iloc[-2] and \
                 self.data['short_sma'].iloc[-1] < self.data['long_sma'].iloc[-1]:
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

        recent_data = self.data.tail(self.period)
        highs = recent_data['high']
        lows = recent_data['low']

        # Simplified trendline: a line connecting the highest high and lowest low
        resistance_slope = (highs.max() - highs.iloc[0]) / self.period
        resistance_intercept = highs.iloc[0]
        support_slope = (lows.min() - lows.iloc[0]) / self.period
        support_intercept = lows.iloc[0]

        current_price = self.data['close'].iloc[-1]
        
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

        recent_data = self.data.tail(self.period)
        support_level = recent_data['low'].min()
        resistance_level = recent_data['high'].max()

        current_price = self.data['close'].iloc[-1]
        current_open = self.data['open'].iloc[-1]
        
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

if __name__ == '__main__':
    main()